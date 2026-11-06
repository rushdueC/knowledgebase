# -*- coding: utf-8 -*-
"""
②·补 生成标准答案（reference）—— 解锁 Context Recall / Context Precision 的关键一步

为什么要这步：
    Ragas 四个指标里，context_recall 和 context_precision 需要「标准答案」做参照。
    没有它，你只能评生成（有没有编、有没有答到点上），评不了检索。

本脚本的做法（半自动，不是纯手工）：
    让【另一个模型】读「问题 + 检索到的原文」→ 起草标准答案 → 写入 xlsx
    → 你在 Excel 里过一遍、改掉不对的 → 跑 run_eval.py

⚠️ 为什么可以用模型起草，但必须人工确认：
    · 起草模型 ≠ 生成模型（默认 qwen3:14b vs qwen3:8b），不是自我循环
    · 但模型可能写错、写漏、把无关内容写进去 —— 所以必须人工过一遍
    · 尤其是「文档未提及」类的题，模型经常自作主张给个答案

用法：
    python make_reference.py            # 给当前 VERSION 的数据集补 reference
    python make_reference.py v2         # 指定版本
    python make_reference.py --force    # 已填过的也重新生成（慎用）
"""
import sys
import requests
import pandas as pd
from pathlib import Path

from config import (DATA_DIR, DATA_OUT, OLLAMA_URL, REF_MODEL, GEN_MODEL,
                    NUM_CTX, VERSION, REFERENCE_PROMPT, SIM_THRESHOLD,
                    REF_TOP_K, TOP_K)

# 门禁一律用【余弦相似度】，和开不开 rerank 无关。
# cross-encoder 分数是另一个量纲（实测相关句 0.05 就算高、无关句 0.000），
# 拿 0.45 这种余弦阈值去卡它会把正常题全判成「文档未提及」。
# run_batch 会在数据集里同时写 top1_similarity（排序分数）和
# top1_vector_similarity（余弦），这里优先用后者。
DRAFT_FLAG = "ref_is_draft"      # 标记列：True = 机器起草，待人工确认


def strip_think(text: str) -> str:
    """去掉 qwen3 思考模式残留的 <think>...<think>"""
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    return text.strip()


def gen_reference(contexts, question):
    ctx = "\n\n---\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    prompt = REFERENCE_PROMPT.format(contexts=ctx, question=question)

    payload = {
        "model": REF_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": NUM_CTX},
    }
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=900)
    r.raise_for_status()
    return strip_think(r.json()["message"]["content"])


def main():
    args = [a for a in sys.argv[1:]]
    force = "--force" in args
    ver = next((a for a in args if not a.startswith("--")), VERSION)

    f = DATA_DIR / f"eval_data_{ver}.xlsx"
    if not f.exists():
        print(f"❌ 找不到 {f}")
        print(f"   现有：{[p.name for p in DATA_DIR.glob('eval_data_*.xlsx')]}")
        print(f"   👉 请先运行 python run_batch.py")
        return

    df = pd.read_excel(f)
    print(f"✅ 读取 {f.name}：{len(df)} 条")

    if "reference" not in df.columns:
        df["reference"] = ""
    if DRAFT_FLAG not in df.columns:
        df[DRAFT_FLAG] = False

    # ⚠️ pandas 3.x 的坑：reference 列全是空值时会被推断成 float64，
    #    往里写字符串会直接抛 pandas.errors.LossySetitemError
    #    （旧版 pandas 会自动放宽成 object，所以这问题以前没暴露）。
    #    这里统一把要写的列强制成 object，并把 NaN 换成空串。
    for _col in ("reference", "response", "user_input", "retrieved_contexts"):
        if _col in df.columns:
            df[_col] = df[_col].astype("object")
    df["reference"] = df["reference"].where(df["reference"].notna(), "")
    df[DRAFT_FLAG] = df[DRAFT_FLAG].astype("object")

    def filled(x):
        s = str(x).strip()
        return s != "" and s.lower() != "nan"

    # ---------- 安全检查：起草模型不能等于生成模型 ----------
    if REF_MODEL == GEN_MODEL:
        print(f"⚠️ 警告：REF_MODEL 和 GEN_MODEL 都是 {GEN_MODEL}")
        print("   这等于让被评测的选手自己出标准答案，分数会虚高。")
        print("   建议在 config.py 里把 REF_MODEL 改成一个更大的模型。\n")

    todo = []
    for i, row in df.iterrows():
        already = filled(row.get("reference", ""))
        if already and not force:
            continue          # 已填过就跳过；--force 时全部重生成
        todo.append(i)

    if not todo:
        print("✅ reference 列已经填满了，无需生成")
        print(f"   👉 下一步直接跑：python run_eval.py")
        return

    print(f"⏳ 需要生成 {len(todo)} 条，起草模型 = {REF_MODEL}\n")

    # ⭐ 用【更大】的 top_k 重新检索来起草标准答案。
    # 旧版直接用 xlsx 里的 retrieved_contexts（= 正式评测的 TOP_K=5），
    # 于是 reference 的每句话必然在这 5 段里 → context_recall 恒等于 1.0，
    # 指标彻底失效（config.py 的 REF_TOP_K 注释写了这事，但代码从没实现）。
    wide_k = max(int(REF_TOP_K or 0), int(TOP_K or 0))
    use_wide = wide_k > int(TOP_K or 0)
    if use_wide:
        print(f"🔎 起草时用 REF_TOP_K={wide_k} 重新检索"
              f"（正式评测只取 TOP_K={TOP_K}），"
              f"这样 context_recall 才有区分度")
    else:
        print(f"⚠️ REF_TOP_K({REF_TOP_K}) 不大于 TOP_K({TOP_K})："
              f"将直接用评测上下文起草，context_recall 会恒等于 1.0")
    print()

    if use_wide:
        try:
            from run_batch import retrieve_with_rerank
        except Exception as e:                      # noqa: BLE001
            print(f"⚠️ 无法导入检索模块（{e}），退回使用 xlsx 里的上下文")
            use_wide = False


    low_sim_warn = []
    for n, i in enumerate(todo, 1):
        q = str(df.at[i, "user_input"]).strip()
        ctxs = [c.strip() for c in str(df.at[i, "retrieved_contexts"]).split("|||")
                if c.strip() and c.strip() != "nan"]

        # 门禁分数：优先用余弦（top1_vector_similarity），没有就用旧列的 top1_similarity
        sim_col = ("top1_vector_similarity"
                   if "top1_vector_similarity" in df.columns else "top1_similarity")
        sim = df.at[i, sim_col] if sim_col in df.columns else None
        try:
            sim_f = float(sim)
        except (TypeError, ValueError):
            sim_f = None

        # 起草用更宽的上下文（见上面的说明）。
        # 用了宽检索就以宽检索的【余弦】top1 为准，避免拿旧分数卡新上下文。
        if use_wide:
            try:
                wide_docs, _, _, wide_vec = retrieve_with_rerank(q, k=wide_k)
                if wide_docs:
                    ctxs = wide_docs
                    if wide_vec is not None:
                        sim_f = float(wide_vec)
            except Exception as e:                  # noqa: BLE001
                print(f"   ⚠️ 宽检索失败，用原上下文：{type(e).__name__}: {e}")

        if not ctxs:
            df.at[i, "reference"] = "文档未提及"
            df.at[i, DRAFT_FLAG] = True
            print(f"[{n}/{len(todo)}] ⚠️ 无上下文 → 直接标为「文档未提及」| {q[:30]}")
            continue

        # 余弦相似度低于阈值 → 判定为「文档里没有答案」，不调模型起草
        flag = ""
        if sim_f is not None and sim_f < SIM_THRESHOLD:
            df.at[i, "reference"] = "文档未提及"
            df.at[i, DRAFT_FLAG] = True
            print(f"[{n}/{len(todo)}] 🔍 余弦 {sim_f:.3f} < {SIM_THRESHOLD} "
                  f"→ 判为「文档未提及」| {q[:26]}")
            low_sim_warn.append((q, sim_f, "文档未提及"))
            continue

        try:
            ref = gen_reference(ctxs, q)
        except Exception as e:
            print(f"[{n}/{len(todo)}] ❌ 生成失败：{e}")
            continue

        # 起草出来是拒答措辞的，也统一成「文档未提及」
        if "未提及" in ref or "无法回答" in ref or len(ref) < 6:
            ref = "文档未提及"

        # 刚过阈值的，提醒人工重点看
        if sim_f is not None and sim_f < SIM_THRESHOLD * 1.2:
            flag = " ⚠️相似度偏低，请人工确认"
            low_sim_warn.append((q, sim_f, ref))

        df.at[i, "reference"] = ref
        df.at[i, DRAFT_FLAG] = True
        print(f"[{n}/{len(todo)}] ✅ {q[:28]} → {ref[:50]}{flag}")

    DATA_DIR.mkdir(exist_ok=True)
    df.to_excel(f, index=False)
    print(f"\n💾 已写入 {f}")

    # ---------- 人工确认清单 ----------
    print("\n" + "═" * 70)
    print("【务必人工过一遍】")
    print(f"   打开 {f}")
    print(f"   看 reference 列，重点检查：")
    print(f"     1. 答案是不是真的来自你的文档（模型可能编）")
    print(f"     2. 有没有漏掉要点（会导致 context_recall 偏低）")
    print(f"     3. 文档里没答案的题，reference 应该是「文档未提及」")
    print(f"   改完后把 {DRAFT_FLAG} 列改成 False（或删掉该列）")

    if low_sim_warn:
        print(f"\n   ⚠️ {len(low_sim_warn)} 条相似度偏低，标准答案可能不可靠：")
        for q, s, r in low_sim_warn:
            print(f"     - {q[:30]} (sim={s:.3f}) → {r[:40]}")

    print("\n👉 确认完成后：python run_eval.py")


if __name__ == "__main__":
    main()
