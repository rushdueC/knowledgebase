# -*- coding: utf-8 -*-
"""
④ 多版本对比：把 data/scores_v*.csv 汇总成一张表，看调参到底有没有用

用法：python compare.py
"""
import json
import glob
import pandas as pd
from pathlib import Path

from config import DATA_DIR

# ragas 各版本列名略有出入，这里都列出来
METRIC_ALIASES = {
    "faithfulness":      ["faithfulness"],
    "answer_relevancy":  ["answer_relevancy", "response_relevancy"],
    "context_recall":    ["context_recall", "llm_context_recall"],
    "context_precision": ["llm_context_precision_with_reference",
                          "context_precision", "llm_context_precision"],
}


def pick(df, key):
    for c in METRIC_ALIASES[key]:
        if c in df.columns:
            return c
    return None


def main():
    score_files = sorted(glob.glob(str(DATA_DIR / "scores_v*.csv")))
    if not score_files:
        print(f"❌ {DATA_DIR} 下没有 scores_v*.csv")
        return

    rows = []
    for f in score_files:
        ver = Path(f).stem.replace("scores_", "")
        df = pd.read_csv(f)
        row = {"版本": ver}

        for key in METRIC_ALIASES:
            c = pick(df, key)
            if c and pd.api.types.is_numeric_dtype(df[c]):
                row[key] = round(float(df[c].mean()), 4)

        # 附上这版的参数
        mp = DATA_DIR / f"eval_data_{ver}.json"
        if mp.exists():
            meta = json.loads(mp.read_text(encoding="utf-8"))
            row["chunk"] = meta.get("CHUNK_SIZE")
            row["top_k"] = meta.get("TOP_K")
            row["生成模型"] = meta.get("GEN_MODEL")
            row["嵌入模型"] = meta.get("EMBED_MODEL")
        rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["版本"] + [c for c in METRIC_ALIASES if c in out.columns] + \
           [c for c in ["chunk", "top_k", "生成模型", "嵌入模型"] if c in out.columns]
    print("\n===== 各版本对比 =====")
    print(out[cols].to_string(index=False))

    if len(rows) >= 2:
        base, last = rows[0], rows[-1]
        print(f"\n===== {last['版本']} 相对 {base['版本']} 的变化 =====")
        for key in METRIC_ALIASES:
            if key in base and key in last:
                b, l = base[key], last[key]
                if pd.isna(b) or pd.isna(l):
                    continue
                d = l - b
                flag = "🔺" if d > 0.02 else ("🔻" if d < -0.02 else "➖")
                print(f"  {flag} {key:18s} {b:.4f} → {l:.4f}  ({d:+.4f})")

        print("\n⚠️ 本地小评委波动可达 ±0.2，变化小于 0.05 的不要当真，"
              "只信多次一致的大幅趋势。")

    out.to_csv(DATA_DIR / "compare.csv", index=False, encoding="utf-8-sig")
    print(f"\n💾 对比表已保存 → {DATA_DIR / 'compare.csv'}")


if __name__ == "__main__":
    main()
