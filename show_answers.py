# -*- coding: utf-8 -*-
"""
👀 查看完整问答：把 data/eval_data_vX.xlsx 里的问答整段打印出来
用法：
    python show_answers.py          # 默认看 config.py 里的 VERSION
    python show_answers.py v2       # 看指定版本
    python show_answers.py v1 3     # 只看第 3 条
"""
import sys
import textwrap
import pandas as pd
from pathlib import Path

from config import DATA_DIR, VERSION, SIM_THRESHOLD

# 门禁一律用余弦相似度：run_batch 会写 top1_vector_similarity（永远余弦），
# 只有老数据集才没有这列，那时退回 top1_similarity。
SIM_COL_CANDIDATES = ("top1_vector_similarity", "top1_similarity")


def wrap(s, width=92, indent="    "):
    out = []
    for line in str(s).splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(textwrap.fill(line, width=width,
                                 initial_indent=indent, subsequent_indent=indent))
    return "\n".join(out) if out else indent + "（空）"


def main():
    ver = sys.argv[1] if len(sys.argv) > 1 else VERSION
    only = int(sys.argv[2]) if len(sys.argv) > 2 else None

    f = DATA_DIR / f"eval_data_{ver}.xlsx"
    if not f.exists():
        print(f"❌ 找不到 {f}")
        print(f"   现有文件：{[p.name for p in DATA_DIR.glob('eval_data_*.xlsx')]}")
        return

    df = pd.read_excel(f)
    print(f"📖 {f.name} —— 共 {len(df)} 条\n")

    items = [df.iloc[only - 1]] if only else [df.iloc[i] for i in range(len(df))]
    start = only - 1 if only else 0

    for n, row in enumerate(items, start + 1):
        q = str(row["user_input"]).strip()
        ctxs = [c.strip() for c in str(row.get("retrieved_contexts", "")).split("|||")
                if c.strip() and c.strip() != "nan"]
        resp = str(row.get("response", "")).strip()
        if resp == "nan":
            resp = ""
        ref = str(row.get("reference", "")).strip()
        if ref == "nan":
            ref = ""
        sim = row.get("top1_similarity", None)

        print("═" * 96)
        print(f"【{n}】{q}")
        sim_str = f"  (最高相似度 {sim:.3f})" if pd.notna(sim) else ""
        print(f"    {len(ctxs)} 段上下文 | 回答 {len(resp)} 字{sim_str}")
        print("─" * 96)

        print("\n  ▸ 检索到的上下文：")
        for i, c in enumerate(ctxs, 1):
            print(f"    [{i}] {wrap(c, indent='       ').lstrip()[:600]}")
            if len(c) > 600:
                print("       ...")

        print("\n  ▸ 模型回答：")
        print(wrap(resp))

        print("\n  ▸ 标准答案：")
        print(wrap(ref) if ref else "    （未填）")
        print()

    # ---------- 汇总 ----------
    print("═" * 96)
    print("【汇总】")
    lens = [len(str(r).strip()) if str(r).strip() != "nan" else 0
            for r in df["response"]]
    print(f"  回答长度：最短 {min(lens)} 字 / 最长 {max(lens)} 字 / 平均 {sum(lens)/len(lens):.0f} 字")

    short = df[[l < 60 for l in lens]]
    if len(short):
        print(f"\n  ⚠️ {len(short)} 条回答过短（<60 字），这些大概率是拒答或检索失败：")
        for _, r in short.iterrows():
            print(f"     - {str(r['user_input'])[:34]} → {str(r['response'])[:50]}")

    sim_col = next((c for c in SIM_COL_CANDIDATES if c in df.columns), None)
    if sim_col:
        s = df[sim_col].astype(float)
        print(f"\n  余弦相似度：最低 {s.min():.3f} / 最高 {s.max():.3f} / "
              f"平均 {s.mean():.3f}")
        low = df[s < SIM_THRESHOLD]
        if len(low):
            print(f"  ⚠️ {len(low)} 条余弦 < {SIM_THRESHOLD}，检索不准：")
            for _, r in low.iterrows():
                print(f"     - {str(r['user_input'])[:34]} → {r[sim_col]:.3f}")


if __name__ == "__main__":
    main()
