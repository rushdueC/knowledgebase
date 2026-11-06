# -*- coding: utf-8 -*-
"""
管线体检 —— 逐段测试，定位「卡在哪一步」

为什么需要：
    界面卡住时你只看到「检索中...」，根本不知道是
      · Ollama 没启动
      · embedding 模型没拉/没加载
      · 向量库不存在或为空
      · 生成模型没拉/慢
      · 还是 rerank 在下载
    本脚本把每一步拆开单独测，每步都有短超时，【保证不会整体卡死】。

用法：
    python check_pipeline.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

BASE = Path(__file__).resolve().parent
STEP_TIMEOUT = 20      # 常规步骤最多等 20 秒
# 生成单独放宽：qwen3:8b 首次调用要把约 5GB 模型加载进内存，
# 20s 根本不够，会【误判成失败】。设 120s 才能区分「慢」和「挂」。
GEN_TIMEOUT = 120

results = []


def mark(ok, name, detail=""):
    results.append((ok, name, detail))
    flag = "✅" if ok else "❌"
    print(f"  {flag} {name}" + (f"  —— {detail}" if detail else ""))
    return ok


def run_with_timeout(fn, timeout, label):
    """子线程跑 fn，超时返回 (None, 超时信息)"""
    import threading
    box = {}

    def target():
        try:
            box["v"] = fn()
        except BaseException as e:      # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, f"超过 {timeout}s 未完成"
    if "e" in box:
        return None, f"{type(box['e']).__name__}: {box['e']}"
    return box.get("v"), None


def main():
    # ---------- 读 config ----------
    print("=" * 64)
    print("① 配置文件")
    print("=" * 64)
    try:
        import config
        mark(True, "config.py 可导入")
    except Exception as e:
        mark(False, "config.py 导入失败", str(e))
        return

    OLLAMA = getattr(config, "OLLAMA_URL", "http://localhost:11434")
    EMBED = getattr(config, "EMBED_MODEL", "")
    GEN = getattr(config, "GEN_MODEL", "")
    JUDGE = getattr(config, "JUDGE_MODEL", "")
    DB_DIR = getattr(config, "DB_DIR", BASE / "vector_store")
    COLL = getattr(config, "COLLECTION", "")
    RERANK_ON = bool(getattr(config, "RERANK_ENABLED", False))
    RERANK_BACK = getattr(config, "RERANK_BACKEND", "http")
    RERANK_LOCAL = getattr(config, "RERANK_LOCAL_PATH", "") or ""
    RERANK_LOAD_TIMEOUT = int(getattr(config, "RERANK_LOAD_TIMEOUT", 180) or 180)
    DISABLE_THINK = bool(getattr(config, "DISABLE_THINK", True))

    print(f"  OLLAMA_URL  = {OLLAMA}")
    print(f"  EMBED_MODEL = {EMBED}")
    print(f"  GEN_MODEL   = {GEN}")
    print(f"  JUDGE_MODEL = {JUDGE}")
    print(f"  COLLECTION  = {COLL}")
    print(f"  rerank      = {'开' if RERANK_ON else '关'} "
          f"（后端 {RERANK_BACK}）")
    if RERANK_ON and RERANK_BACK == "local":
        if RERANK_LOCAL and Path(RERANK_LOCAL).exists():
            print(f"  📂 本地模型目录存在：{RERANK_LOCAL}")
        else:
            print(f"  ⚠️ 本地模型还没下载 → rerank 会自动降级（不影响主流程）")
            print(f"     👉 先跑 python predownload_rerank.py")

    # ---------- ② Ollama ----------
    print()
    print("=" * 64)
    print("② Ollama 服务")
    print("=" * 64)

    def check_tags():
        r = requests.get(f"{OLLAMA}/api/tags", timeout=STEP_TIMEOUT)
        r.raise_for_status()
        return r.json()

    data, err = run_with_timeout(check_tags, STEP_TIMEOUT, "tags")
    if err:
        mark(False, "Ollama 不可达", err)
        print("\n  👉 启动 Ollama（桌面版托盘图标 / 或命令行 ollama serve）")
        print(f"     手动验证：curl {OLLAMA}/api/tags")
        return
    mark(True, "Ollama 可达")

    names = [m.get("name", "") for m in data.get("models", [])]
    print(f"  已装模型（{len(names)}）：")
    for n in names:
        print(f"     · {n}")

    embed_ok = any(EMBED in n or n.startswith(EMBED) for n in names)
    gen_ok = any(GEN in n or n.startswith(GEN) for n in names)
    # 允许 tag 省略（qwen3:8b vs qwen3）
    if not embed_ok:
        embed_ok = any(n.split(":")[0] == EMBED.split(":")[0] for n in names)
    if not gen_ok:
        gen_ok = any(n.split(":")[0] == GEN.split(":")[0] for n in names)

    mark(embed_ok, f"嵌入模型 {EMBED} 已安装",
         "" if embed_ok else f"👉 ollama pull {EMBED}")
    mark(gen_ok, f"生成模型 {GEN} 已安装",
         "" if gen_ok else f"👉 ollama pull {GEN}")

    # ---------- ③ 向量库 ----------
    print()
    print("=" * 64)
    print("③ 向量库（chunk 数量在这里看）")
    print("=" * 64)

    db_dir = Path(DB_DIR)
    jsons = sorted(db_dir.glob("*.json")) if db_dir.exists() else []
    if not db_dir.exists():
        mark(False, f"目录不存在：{db_dir}", "👉 python build_index.py")
    elif not jsons:
        mark(False, f"{db_dir} 下没有任何 .json", "👉 python build_index.py")
    else:
        print(f"  找到 {len(jsons)} 个库文件：")
        for j in jsons:
            try:
                d = json.loads(j.read_text(encoding="utf-8"))
                n = len(d.get("chunks", []))
                npy = db_dir / f"{j.stem}.npy"
                dim = ""
                if npy.exists():
                    arr = np.load(npy)
                    dim = f" 维度{arr.shape[1]}" if arr.ndim == 2 else ""
                cur = " ← 当前使用" if j.stem == COLL else ""
                print(f"     · {j.stem}: {n} 块{dim}{cur}")
                if n == 0:
                    print(f"       ⚠️ 空库！")
            except Exception as e:
                print(f"     · {j.stem}: 读取失败 {e}")

        target = db_dir / f"{COLL}.json"
        if not target.exists():
            mark(False, f"当前 COLLECTION 的库不存在：{target.name}",
                 "👉 python build_index.py")
        else:
            d = json.loads(target.read_text(encoding="utf-8"))
            n = len(d.get("chunks", []))
            mark(n > 0, f"当前库 {COLL} 共 {n} 块",
                 "" if n > 0 else "空库，需重建")
            if n < 10:
                print(f"     ⚠️ 只有 {n} 块，检索效果会很差")
                print(f"        多半是 docs/ 里的文档没被读到（缺 pypdf 等包）")

    # ---------- ④ embedding ----------
    print()
    print("=" * 64)
    print("④ 单次 embedding（检索慢通常卡在这）")
    print("=" * 64)

    def do_embed():
        r = requests.post(f"{OLLAMA}/api/embed",
                          json={"model": EMBED, "input": ["测试"]},
                          timeout=STEP_TIMEOUT)
        if r.status_code == 200:
            return ("embed", r.json()["embeddings"][0])
        r2 = requests.post(f"{OLLAMA}/api/embeddings",
                           json={"model": EMBED, "prompt": "测试"},
                           timeout=STEP_TIMEOUT)
        r2.raise_for_status()
        return ("embeddings", r2.json()["embedding"])

    t0 = time.perf_counter()
    vec, err = run_with_timeout(do_embed, STEP_TIMEOUT, "embed")
    el = time.perf_counter() - t0
    if err:
        mark(False, "embedding 调用失败", err)
    else:
        kind, v = vec
        mark(True, f"embedding 成功（{el*1000:.0f}ms，{len(v)} 维，端点 /api/{kind}）")
        if el > 5:
            print(f"     ⚠️ 偏慢（{el:.1f}s），首次调用要加载模型属正常")

    # ---------- ⑤ 生成 ----------
    print()
    print("=" * 64)
    print("⑤ 单次生成（非流式，短问题）")
    print("=" * 64)

    def do_gen():
        # qwen3 默认开启「思考模式」，会在回答前先输出一大段 <think>，
        # 测试时加官方的 /no_think 关闭，否则单纯测个连通性就要几十秒。
        content = "只回复：OK"
        if DISABLE_THINK:
            content += "\n/no_think"
        r = requests.post(f"{OLLAMA}/api/chat", json={
            "model": GEN, "stream": False,
            "messages": [{"role": "user", "content": content}],
            "options": {"temperature": 0, "num_ctx": 2048},
        }, timeout=GEN_TIMEOUT)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    print(f"  ⏱ 本次超时 {GEN_TIMEOUT}s"
          f"（首次调用含模型加载，20s 太短会误判，已单独放宽）")
    t0 = time.perf_counter()
    out, err = run_with_timeout(do_gen, GEN_TIMEOUT, "gen")
    el = time.perf_counter() - t0
    if err:
        mark(False, "生成失败", err)
        print(f"\n  👉 若这是【第一次】调用，很可能只是模型加载慢（8B 约 5GB）。")
        print(f"     先在命令行预热一次，把模型留在内存里：")
        print(f"       ollama run {GEN} \"你好\"")
        print(f"     然后再跑本脚本，通常就秒回了。")
        print(f"\n     想让界面彻底不慢，可换更小的模型：")
        print(f"       config.py: GEN_MODEL = \"qwen3:4b\"")
    else:
        mark(True, f"生成成功（{el:.1f}s）→ {out[:40]!r}")
        if el > 10:
            print(f"     ⚠️ 偏慢（{el:.1f}s）。")
            print(f"        首次调用含模型加载属正常；之后应在数秒内。")
            print(f"        持续慢 → 8B 在 CPU 上跑就是这样，")
            print(f"        可把 GEN_MODEL 换成 qwen3:4b 或云端模型")

    # ---------- ⑥ rerank ----------
    print()
    print("=" * 64)
    print("⑥ rerank（可选增强，失败会自动降级）")
    print("=" * 64)

    if not RERANK_ON:
        print("  ℹ️ rerank 关闭（config.py: RERANK_ENABLED = False）")
    else:
        def do_rerank():
            import rerank as _rk
            rk = _rk.Reranker()
            out, scores = rk.rerank(
                "战略规划模型有哪些",
                ["SWOT分析是一种战略工具", "今天天气不错"], top_n=2)
            return rk, out, scores

        rk_timeout = max(RERANK_LOAD_TIMEOUT, 60) + 30
        got, err = run_with_timeout(do_rerank, rk_timeout, "rerank")
        if err:
            mark(False, "rerank 调用异常", err)
        else:
            rk, out, scores = got
            if scores is None:
                mark(False, f"rerank 未生效（{RERANK_BACK}）",
                     (rk.fail_reason or "").splitlines()[0][:80])
                if RERANK_BACK == "local":
                    print(f"     👉 先预下载：python predownload_rerank.py")
                print(f"     （不影响主流程，会自动降级为纯向量检索）")
            else:
                mark(True, f"rerank 生效（{RERANK_BACK}）",
                     f"相关句 {scores[0]:.4f} / 无关句 {scores[1]:.4f}"
                     if len(scores) > 1 else "")

    # ---------- ⑦ 评测依赖 ----------
    print()
    print("=" * 64)
    print("⑦ 评测依赖（run_eval.py 用）")
    print("=" * 64)

    missing_pkgs = []
    ragas_note = ""
    try:
        import ragas_compat
        ok, problems = ragas_compat.check()
        if not ok:
            missing_pkgs = [p.split(":")[0] for p in problems]
            ragas_note = problems[0][:80]
    except Exception as e:                    # noqa: BLE001
        # 老目录里可能还没有 ragas_compat.py
        for pkg in ("ragas", "openai"):
            try:
                __import__(pkg)
            except ImportError:
                missing_pkgs.append(pkg)
        ragas_note = f"ragas_compat 不可用：{type(e).__name__}"

    if missing_pkgs:
        mark(False, f"缺包/导入失败：{', '.join(missing_pkgs)}",
             ragas_note or "pip install ragas==0.4.3 openai -i "
             "https://pypi.tuna.tsinghua.edu.cn/simple")
    else:
        mark(True, "ragas / openai 可用（评委走 Ollama 的 /v1）")

    judge_ok = any(JUDGE in n or n.split(":")[0] == JUDGE.split(":")[0]
                   for n in names) if JUDGE else False
    if JUDGE and not judge_ok:
        mark(False, f"评委模型 {JUDGE} 不在 ollama list 里",
             f"👉 ollama pull {JUDGE}，或把 config.py 的 JUDGE_MODEL "
             f"换成已装的模型")
    elif JUDGE:
        mark(True, f"评委模型 {JUDGE} 已在 Ollama 里")

    # ---------- ⑧ 结论 ----------
    print()
    print("=" * 64)
    print("【结论】")
    print("=" * 64)
    bad = [r for r in results if not r[0]]
    if not bad:
        print("  ✅ 全链路正常。若界面仍卡，多半是【首次加载模型很慢】，")
        print("     等 1~3 分钟；或在 config.py 把 GEN_MODEL 换成云端模型。")
    else:
        print(f"  ❌ 有 {len(bad)} 项未通过：")
        for _, name, detail in bad:
            print(f"     · {name}" + (f" —— {detail}" if detail else ""))
    print("=" * 64)


if __name__ == "__main__":
    main()
