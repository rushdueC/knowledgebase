# -*- coding: utf-8 -*-
"""
🔍 数据体检：跑 run_eval.py 之前先跑这个，1 秒定位数据哪一环断了
用法：python check_data.py
"""
import re
import pandas as pd
from pathlib import Path

from config import DATA_DIR, DATA_OUT, DB_DIR, COLLECTION, DOCS_DIR, QUESTIONS

REJECT_PAT = [
    "无法回答", "没有提供", "未提及", "不知道", "没有相关",
    "无法确定", "资料中", "cannot", "sorry", "抱歉",
]


def main():
    # ---------- 0. 环境与向量库 ----------
    print("=" * 60)
    print("【0】环境检查")
    vec = DB_DIR / f"{COLLECTION}.npy"
    if vec.exists():
        import numpy as np
        arr = np.load(vec)
        print(f"  ✅ 向量库存在：{arr.shape[0]} 块，维度 {arr.shape[1]}")
    else:
        print(f"  ❌ 向量库不存在：{vec}")
        print(f"     👉 请先运行 python build_index.py")

    doc_files = [p for p in DOCS_DIR.rglob("*")
                 if p.is_file() and p.suffix.lower() in (".txt", ".md", ".pdf", ".docx")]
    print(f"  📁 docs/ 下文档数：{len(doc_files)}")
    for p in doc_files[:10]:
        print(f"     - {p.name}")
    if not doc_files:
        print("     ⚠️ 没有文档！检索必然为空")

    # ---------- 1. 数据集 ----------
    print("=" * 60)
    print("【1】数据集检查")
    if not DATA_OUT.exists():
        print(f"  ❌ 找不到 {DATA_OUT}")
        print(f"     👉 请先运行 python run_batch.py")
        return

    df = pd.read_excel(DATA_OUT)
    print(f"  ✅ {DATA_OUT.name}：{len(df)} 行，列名={list(df.columns)}")

    required = ["user_input", "retrieved_contexts", "response", "reference"]
    for c in required:
        if c not in df.columns:
            print(f"  ❌ 缺列：{c}")
            return

    # ---------- 2. 逐条体检 ----------
    print("=" * 60)
    print("【2】逐条体检\n")

    n_empty_ctx = n_empty_resp = n_reject = 0
    for i, row in df.iterrows():
        q = str(row["user_input"]).strip()
        ctxs = [c.strip() for c in str(row["retrieved_contexts"]).split("|||")
                if c.strip() and c.strip() != "nan"]
        resp = str(row["response"]).strip()
        ref = str(row["reference"]).strip()
        if resp == "nan":
            resp = ""
        if ref == "nan":
            ref = ""

        ctx_len = sum(len(c) for c in ctxs)
        if not ctxs:
            n_empty_ctx += 1
        if not resp:
            n_empty_resp += 1
        is_reject = any(k in resp for k in REJECT_PAT) and len(resp) < 60
        if is_reject:
            n_reject += 1

        flag = "⚠️" if (not ctxs or not resp or is_reject) else "✅"
        print(f"{flag} [{i+1}] {q[:40]}")
        print(f"    上下文：{len(ctxs)} 段 / {ctx_len} 字")
        if ctxs:
            print(f"    片段1：{ctxs[0][:100]}...")
        print(f"    回答  ：{len(resp)} 字 | {resp[:100] if resp else '（空）'}")
        print(f"    标准答案：{ref[:60] if ref else '（未填）'}")
        print()

    # ---------- 3. 汇总诊断 ----------
    print("=" * 60)
    print("【3】诊断结论")
    print(f"  上下文为空的条数：{n_empty_ctx} / {len(df)}")
    print(f"  回答为空的条数  ：{n_empty_resp} / {len(df)}")
    print(f"  疑似拒答的条数  ：{n_reject} / {len(df)}")

    if n_empty_ctx == len(df):
        print("\n  🔴 检索完全没工作 → 检查 docs/ 里有没有文档，重跑 build_index.py")
    elif n_empty_resp == len(df):
        print("\n  🔴 回答全是空的 → run_batch.py 生成环节失败，检查 Ollama 里有没有 qwen3:8b")
    elif n_reject == len(df):
        print("\n  🟠 全部拒答 → 检索到的内容跟问题无关")
        print("     常见原因：embedding 模型是英文的，中文检索失效")
        print("     解法：config.py 里 EMBED_MODEL 改成 qwen3-embedding:0.6b 后重建库")
    elif n_empty_ctx or n_empty_resp:
        print("\n  🟠 部分样本有问题 → 看上面标 ⚠️ 的条目")
    else:
        print("\n  🟢 数据看起来正常，可以跑 run_eval.py 了")

    if df["reference"].isna().all() or (df["reference"].astype(str).str.strip() == "").all():
        print("\n  ℹ️  reference 列全空 → 只会跑 2 个指标")
        print("     想评检索质量（Context Recall/Precision）请把标准答案补上")

    if DATA_OUT.with_suffix(".json").exists():
        import json
        meta = json.loads(DATA_OUT.with_suffix(".json").read_text(encoding="utf-8"))
        print("\n【4】本版参数")
        for k, v in meta.items():
            print(f"     {k} = {v}")


if __name__ == "__main__":
    main()
