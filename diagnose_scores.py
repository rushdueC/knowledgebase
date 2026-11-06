# -*- coding: utf-8 -*-
"""
🔬 分数诊断：把分数和"回答长度/相似度/上下文"放一起看，判断高分是真还是虚

用法：python diagnose_scores.py v1
"""
import sys
import json
import pandas as pd
from pathlib import Path

from config import DATA_DIR, VERSION, SIM_THRESHOLD

# 门禁一律用余弦相似度（run_batch 写的 top1_vector_similarity）；
# 老数据集没有这列时退回 top1_similarity。
SIM_COL_CANDIDATES = ("top1_vector_similarity", "top1_similarity")


def main():
    ver = sys.argv[1] if len(sys.argv) > 1 else VERSION
    dx = DATA_DIR / f"eval_data_{ver}.xlsx"
    ds = DATA_DIR / f"scores_{ver}.csv"

    if not dx.exists():
        print(f"❌ 找不到 {dx}")
        return
    if not ds.exists():
        print(f"❌ 找不到 {ds}，先跑 python run_eval.py")
        return

    df = pd.read_excel(dx)
    sc = pd.read_csv(ds)

    # 按 user_input 对齐（分数表行序通常与数据集一致）
    if len(sc) == len(df):
        for c in sc.columns:
            if c not in df.columns:
                df[c] = sc[c].values
    else:
        print(f"⚠️ 数据集 {len(df)} 行 / 分数 {len(sc)} 行，行数不一致，仅看分数表")
        df = sc.copy()

    def n(x):
        s = str(x).strip()
        return 0 if s in ("", "nan", "None") else len(s)

    df["回答字数"] = df["response"].apply(n)
    df["上下文段数"] = df["retrieved_contexts"].apply(
        lambda x: len([c for c in str(x).split("|||") if c.strip() and c.strip() != "nan"]))
    df["上下文字数"] = df["retrieved_contexts"].apply(
        lambda x: sum(len(c.strip()) for c in str(x).split("|||")
                      if c.strip() and c.strip() != "nan"))

    sim_col = next((c for c in SIM_COL_CANDIDATES if c in df.columns), None)
    show = ["user_input", "回答字数", "上下文字数", sim_col,
            "faithfulness", "answer_relevancy"]
    show = [c for c in show if c and c in df.columns]

    print("═" * 104)
    print("【分数 × 数据对照】")
    print(df[show].to_string(index=False))

    # ---------- 满分可信度分析 ----------
    print("\n" + "═" * 104)
    print("【满分可信度分析】")

    if "faithfulness" in df.columns:
        full = df[df["faithfulness"].fillna(-1) >= 0.999]
        print(f"\n  faithfulness = 1.0 的条数：{len(full)} / {len(df)}")

        if len(full):
            short = full[full["回答字数"] < 120]
            print(f"  其中回答 <120 字的：{len(short)} 条  ← 这些满分不可信")
            if len(short):
                print("\n  ⚠️ 短回答拿满分 = 评委挑不出几条 claim，"
                      "分母太小，1.0 属于虚高：")
                for _, r in short.iterrows():
                    resp = str(r["response"]).strip().replace("\n", " ")
                    print(f"     - {str(r['user_input'])[:26]} | {r['回答字数']}字 | {resp[:60]}")

            long_full = full[full["回答字数"] >= 120]
            if len(long_full):
                print(f"\n  ✅ {len(long_full)} 条是长回答拿满分，这个满分含金量高")

    # ---------- 相似度 vs 忠实度 ----------
    if sim_col and "faithfulness" in df.columns:
        s = df[sim_col].astype(float)
        print(f"\n  余弦相似度：最低 {s.min():.3f} / 平均 {s.mean():.3f} / "
              f"最高 {s.max():.3f}")
        low = df[s < SIM_THRESHOLD]
        if len(low) and len(low) < len(df):
            print(f"  ⚠️ {len(low)} 条余弦 <{SIM_THRESHOLD} 却仍被打分 "
                  f"→ 检索不准时分数无意义：")
            for _, r in low.iterrows():
                print(f"     - {str(r['user_input'])[:30]} "
                      f"sim={r[sim_col]:.3f} faith={r.get('faithfulness','-')}")

    # ---------- 结论 ----------
    print("\n" + "═" * 104)
    print("【结论】")

    if "faithfulness" in df.columns:
        m = df["faithfulness"].mean()
        if m >= 0.95:
            print("  🟡 faithfulness 已饱和（≥0.95），失去区分能力。")
            print("     原因：你的 system prompt 强制「只用参考资料」，模型几乎不可能编造，")
            print("     所以满分是【设计使然】，不是评测成功。")
            print("\n     想让它重新变得有区分度，必须加入让模型「想编」的题：")
            print("       ① 文档里根本没有答案的问题（幻觉探针，最重要）")
            print("       ② 需要跨多个片段整合的题（容易漏、容易串）")
            print("       ③ 数字/时间/名称细节题（最容易记错）")
            print("       ④ 两个相似概念的辨析题（容易张冠李戴）")
        elif m >= 0.8:
            print("  🟢 faithfulness 在合理区间，指标有区分度")
        else:
            print("  🔴 faithfulness 偏低，存在明显幻觉，重点查短回答和拒答样本")

    if "answer_relevancy" in df.columns:
        m = df["answer_relevancy"].mean()
        print(f"\n  answer_relevancy = {m:.3f}")
        if m < 0.8:
            print("     🟠 偏低 → 回答没答到点上。常见原因：")
            print("        · 检索到的片段跟问题不相关（看上面相似度）")
            print("        · 回答太短/太泛，评委反推不出问题")
            print("        · 模型过度保守，老是说「资料中没有」")
        else:
            print("     🟢 回答基本切题")

    # ---------- 检索侧两个指标 ----------
    for key, aliases, cn in [
        ("context_recall", ["context_recall", "llm_context_recall"], "context_recall"),
        ("context_precision", ["llm_context_precision_with_reference",
                               "context_precision", "llm_context_precision"],
         "context_precision"),
    ]:
        col = next((c for c in aliases if c in df.columns), None)
        if not col:
            continue
        m = df[col].mean()
        print(f"\n  {cn} = {m:.3f}")
        if cn == "context_recall":
            if m < 0.7:
                print("     🟠 偏低 → 该检的内容没检到（漏召回）")
                print("        修法：加大 TOP_K，或减小 CHUNK_SIZE")
                print("        ⚠️ 也可能是 reference 写得比文档更详细，"
                      "导致评委认为上下文里缺内容")
            else:
                print("     🟢 该检的基本都检到了")
        else:
            if m < 0.7:
                print("     🟠 偏低 → 相关内容被排在后面（排序差）")
                print("        修法：换更强 embedding、加重排模型、或减小 CHUNK_SIZE")
            else:
                print("     🟢 相关内容排得靠前")

    # ---------- 检索 vs 生成 的归因 ----------
    cr_col = next((c for c in ["context_recall", "llm_context_recall"]
                   if c in df.columns), None)
    cp_col = next((c for c in ["llm_context_precision_with_reference",
                               "context_precision", "llm_context_precision"]
                   if c in df.columns), None)
    if cr_col and cp_col and "faithfulness" in df.columns:
        cr, cp, fa = df[cr_col].mean(), df[cp_col].mean(), df["faithfulness"].mean()
        print("\n【归因】")
        if cr >= 0.75 and cp >= 0.75 and fa < 0.85:
            print("  🔍 检索健康（recall/precision 都高）但 faithfulness 低")
            print("     → 问题在【生成】：换更大生成模型，或收紧 system prompt")
        elif cr < 0.75 and cp < 0.75:
            print("  🔍 检索两个指标都低 → 问题在【检索】")
            print("     → 优先换中文 embedding（收益最大），再调 chunk / top_k")
        elif cp < 0.75 <= cr:
            print("  🔍 能检到但排得乱 → 典型的排序问题")
            print("     → 加重排模型（rerank）收益最大")
        elif cr < 0.75 <= cp:
            print("  🔍 排得准但漏内容 → 召回不全")
            print("     → 加大 TOP_K，或减小 CHUNK_SIZE")
        else:
            print("  🟢 检索与生成都在合理区间")


if __name__ == "__main__":
    main()
