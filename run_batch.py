# -*- coding: utf-8 -*-
"""
② 批量跑题：对每个问题 → 检索原文片段 → 让模型作答 → 存成数据集

这就是替代"手工在界面提问 + 复制粘贴"的那一步。
输出 data/eval_data_v1.xlsx，四列齐全可直接喂给 Ragas。

【无需 chromadb】直接读 numpy 索引。
"""
import json
import requests
import pandas as pd
from pathlib import Path

from config import (BASE, DATA_DIR, DB_DIR, QUESTIONS, OLLAMA_URL, EMBED_MODEL,
                    GEN_MODEL, TOP_K, TEMPERATURE, NUM_CTX, DISABLE_THINK,
                    COLLECTION, CHUNK_SIZE, CHUNK_OVERLAP, DATA_OUT, VERSION,
                    RAG_SYSTEM_PROMPT)

# ---- 新增字段用 getattr 兜底 ----
# 旧版 config.py 没有 RERANK_* 时不报错，rerank 默认关闭
import config as _cfg
RERANK_ENABLED = bool(getattr(_cfg, "RERANK_ENABLED", False))
RETRIEVE_K     = int(getattr(_cfg, "RETRIEVE_K", 20) or 20)

import retrieval as _retrieval

VEC_PATH = DB_DIR / f"{COLLECTION}.npy"
IDX_PATH = DB_DIR / f"{COLLECTION}.json"


# ---------------------------------------------------------
#  检索：统一走 retrieval.Retriever
#  （问答页 / 实验引擎 / run_batch / make_reference 共用同一份实现，
#    参数一处生效，不会出现「前端调了、脚本没变」）
# ---------------------------------------------------------
def embed_one(text):
    """把一句话变成向量（已归一化）"""
    return _retrieval.embed_one(text, EMBED_MODEL, OLLAMA_URL)


def load_index():
    return _retrieval.load_index(VEC_PATH, IDX_PATH)


_RETRIEVER = None


def get_retriever():
    """进程内单例：索引只读一次，rerank 模型只加载一次"""
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = _retrieval.Retriever(
            top_k=TOP_K, pool_k=RETRIEVE_K,
            use_rerank=RERANK_ENABLED,
            use_hybrid=bool(getattr(_cfg, "HYBRID_ENABLED", False)),
            vec_path=VEC_PATH, idx_path=IDX_PATH,
            embed_model=EMBED_MODEL, ollama_url=OLLAMA_URL)
    return _RETRIEVER


def get_reranker():
    """给 main() 打状态用；没开 rerank 时返回 None"""
    return get_retriever().reranker


def retrieve(question, k=TOP_K):
    """纯向量检索（不做 rerank）"""
    return get_retriever().vector_search(question, k)


def retrieve_with_rerank(question, k=TOP_K, pool_k=None):
    """
    两阶段检索：
      ① 向量粗排，取 pool_k 个候选（比 TOP_K 大得多，给 rerank 足够候选）
      ② rerank 精排，取前 k 个

    rerank 失败会自动降级为纯向量检索（rerank.py 内部处理），不中断。

    返回 (docs, metas, scores, vec_top1)：
        scores     —— 实际用于排序的分数（开 rerank 时是 cross-encoder 分数，否则是余弦）
        vec_top1   —— 【永远是余弦相似度】的 top1
                      为什么要单独给：cross-encoder 分数和余弦不是一个量纲
                      （实测相关句 0.05 就算高、无关句 0.000），拿 0.45 这种
                      余弦阈值去卡 rerank 分数会把正常题全判成「没检索到」。
                      所以「这题文档里到底有没有」的门禁一律用 vec_top1。
    """
    res = get_retriever().search(question, top_k=k,
                                 pool_k=pool_k or max(RETRIEVE_K, int(k)))
    return res.docs, res.metas, res.scores, res.vec_top1


def strip_think(text):
    """去掉 qwen3 可能残留在正文里的思考块（防御性处理，正常不会出现）"""
    import re
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def generate(contexts, question):
    ctx = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    user = f"【参考资料】\n{ctx}\n\n【问题】\n{question}"
    if DISABLE_THINK:
        # Qwen3 官方的软开关就是 /no_think。
        # （旧版这里拼了一串来历不明的 <arg_key:...>，模型会把它当正文复述出来，
        #   既没关掉思考，又污染了回答 —— 已删除。）
        user += "\n/no_think"

    payload = {
        "model": GEN_MODEL,
        "messages": [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
    }
    # 1800s：bge-m3 与 qwen3:8b 同时驻留时，单题曾超过 15 分钟（2026-11-06 实测）
    last_err = None
    for attempt in (1, 2):
        try:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=1800)
            r.raise_for_status()
            return strip_think(r.json()["message"]["content"])
        except Exception as e:                     # noqa: BLE001
            last_err = e
            print(f"      ⚠️ 生成失败（第 {attempt} 次）：{type(e).__name__}: {str(e)[:80]}")
    raise last_err


def main():
    if not QUESTIONS.exists():
        QUESTIONS.write_text(
            "有哪些常见的战略规划模型\n"
            "SWOT分析和PESTEL分析怎么结合使用\n"
            "公司没有相关资料的问题（用来测会不会瞎编）\n",
            encoding="utf-8")
        print(f"📝 已生成示例 {QUESTIONS}，请改成你自己的问题后重跑")
        return

    questions = [l.strip() for l in
                 QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]

    rt = get_retriever()
    rk = rt.reranker                      # 没开 rerank 时是 None
    rerank_on = rk is not None
    recall_src = "混合(BM25+向量)" if rt.use_hybrid else "纯向量"
    mode = (f"{recall_src} 候选{RETRIEVE_K} → rerank({rk.backend}) → 取{TOP_K}"
            if rerank_on else f"{recall_src} TOP_K={TOP_K}")
    print(f"共 {len(questions)} 个问题 | 生成模型={GEN_MODEL} | 检索={mode}\n")

    # ---- 断点续跑：把每题结果立刻落盘，重启时跳过已完成 ----
    import json as _json
    partial = DATA_OUT.with_suffix(".partial.jsonl")
    rows = []
    done_q = {}
    if partial.exists():
        for line in partial.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    _r = _json.loads(line)
                    done_q[_r["user_input"]] = _r
                except Exception:                  # noqa: BLE001
                    pass
        if done_q:
            rows = list(done_q.values())
            print(f"♻️ 断点续跑：已有 {len(done_q)} 题完成（{partial.name}），跳过它们\n")

    def _persist(row):
        with open(partial, "a", encoding="utf-8") as f:
            f.write(_json.dumps(row, ensure_ascii=False) + "\n")

    low_sim_cnt = 0
    n_reranked = 0
    # 门禁一律看余弦相似度（vec_top1），和开不开 rerank 无关 ——
    # cross-encoder 分数是另一个量纲，套 0.4 这种阈值会全报警。
    LOW, MID = 0.40, 0.55

    for i, q in enumerate(questions, 1):
        if q in done_q:
            print(f"[{i}/{len(questions)}] ⏭ 已完成，跳过：{q[:24]}")
            continue
        docs, metas, sims, vec_top1 = retrieve_with_rerank(q)
        if rerank_on and not rk.failed:
            n_reranked += 1
        if not docs:
            print(f"[{i}/{len(questions)}] ⚠️ 没检索到内容：{q}")
            answer = ""
        else:
            answer = generate(docs, q)

        rank_top1 = float(sims[0]) if sims else None
        gate = vec_top1 if vec_top1 is not None else (rank_top1 or 0.0)
        rows.append({
            "user_input": q,
            "retrieved_contexts": "|||".join(docs),   # 片段用 ||| 分隔
            "response": answer,
            "reference": "",                          # 留给人工填标准答案
            # 排序用的分数（开 rerank 时是 cross-encoder 分数）
            "top1_similarity": round(rank_top1, 4) if rank_top1 is not None else None,
            # 永远可比的余弦相似度，给 make_reference / 诊断用
            "top1_vector_similarity": round(gate, 4) if vec_top1 is not None else None,
        })

        _persist(rows[-1])          # ← 每题立刻落盘，崩溃不丢进度

        # 分级告警：按余弦相似度判断「检索到底准不准」
        if gate < LOW:
            mark = "⚠️ 相似度偏低"
            low_sim_cnt += 1
        elif gate < MID:
            mark = "🟡 相似度一般"
        else:
            mark = "✅"
        if len(docs) < TOP_K:
            mark += f"(仅召回{len(docs)}段<top_k{TOP_K})"

        extra = (f" | rerank分 {rank_top1:.3f}"
                 if rerank_on and not rk.failed and rank_top1 is not None else "")
        print(f"[{i}/{len(questions)}] {mark} {q[:26]} | 余弦 {gate:.3f}{extra} | "
              f"回答 {len(answer)} 字")

    # ---------- 检索质量汇总：不用等 Ragas 就能判断检索好不好 ----------
    print("\n" + "=" * 55)
    print("【检索质量汇总】")
    print(f"  rerank 状态：{rk.status() if rerank_on else '关闭'}"
          f"（生效 {n_reranked}/{len(questions)} 条）")
    print(f"  余弦相似度 < {LOW} 的问题数：{low_sim_cnt} / {len(questions)}")
    if rerank_on and n_reranked == 0:
        print("  🟠 rerank 开着但一条都没生效 → 已自动降级为纯向量检索")
        print("     先跑一次：python predownload_rerank.py")
        print("     再验证：python rerank.py")
    if low_sim_cnt == len(questions):
        print("  🔴 全部检索失败 → 最可能原因：")
        print("     1) docs/ 里没放真实文档（或只有占位说明文件）")
        print("     2) 用了英文 embedding 处理中文 → 改 config.py 的")
        print("        EMBED_MODEL = 'qwen3-embedding:0.6b' 后重建库")
    elif low_sim_cnt > len(questions) * 0.5:
        print("  🟠 超过一半检索不准 → 优先调 chunk_size 或换 embedding 模型")
    else:
        print("  🟢 检索基本正常，可以跑 run_eval.py")
    print("=" * 55 + "\n")

    DATA_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_excel(DATA_OUT, index=False)
    print(f"\n💾 数据集已保存 → {DATA_OUT}")

    meta = {
        "VERSION": VERSION, "GEN_MODEL": GEN_MODEL, "EMBED_MODEL": EMBED_MODEL,
        "CHUNK_SIZE": CHUNK_SIZE, "CHUNK_OVERLAP": CHUNK_OVERLAP,
        "TOP_K": TOP_K, "TEMPERATURE": TEMPERATURE, "NUM_CTX": NUM_CTX,
        "RERANK_ENABLED": rk.enabled, "RERANK_BACKEND": rk.backend,
        "RERANK_MODEL": rk.model, "RETRIEVE_K": RETRIEVE_K if rk.enabled else None,
    }
    mp = DATA_OUT.with_suffix(".json")
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 参数快照 → {mp}")
    print("\n👉 下一步（二选一）：")
    print("   A. 直接跑：python run_eval.py")
    print("   B. 打开 xlsx 填 reference 列后再跑，可多启用 2 个检索指标")
    print(f"      {DATA_OUT}")


if __name__ == "__main__":
    main()
