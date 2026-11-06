# -*- coding: utf-8 -*-
"""
③ Ragas 打分：用评委模型给 ② 生成的数据集打分

=========================================================
  四大指标（前两个评生成，后两个评检索）
=========================================================
  faithfulness      回答有没有编造          （不需要 reference）
  answer_relevancy  有没有答到点上          （不需要 reference）
  context_recall    该检的是不是漏了        （需要 reference）
  context_precision 检到的是不是排在前面    （需要 reference）

=========================================================
  依赖 —— 只需要两个包
=========================================================
    pip install ragas==0.4.3 openai -i https://pypi.tuna.tsinghua.edu.cn/simple

  评委和向量模型都走 Ollama 的 OpenAI 兼容接口（/v1）：
  用 openai 客户端 + ragas 原生的 llm_factory。
  文件顶部那个 sys.modules 占位补丁是为了让 ragas 能被导入
  （它内部引用了一个在新版依赖里已被移除的模块），细节见 ragas_compat.py。

=========================================================
  前置条件
=========================================================
  只跑前两个 → 直接跑本脚本
  四个全跑   → 先跑 python make_reference.py 生成标准答案，人工确认后再跑本脚本

⚠️ 评委模型（JUDGE_MODEL）必须不同于生成模型（GEN_MODEL），否则自我评分膨胀。
"""
import sys
import warnings
import dataclasses

import pandas as pd


# ---------------------------------------------------------
#  ragas 导入兼容补丁（细节见 ragas_compat.py）
# ---------------------------------------------------------
import ragas_compat

ragas_compat.install()

from config import (DATA_DIR, DATA_OUT, SCORE_OUT, OLLAMA_URL,
                    JUDGE_MODEL, EMBED_MODEL, GEN_MODEL,
                    VERSION, METRICS, EXCLUDE_NO_ANSWER_FROM_RETRIEVAL,
                    EVAL_MAX_RETRIES, EVAL_TIMEOUT, EVAL_MAX_WORKERS,
                    EVAL_RETRY_WAIT, HIDE_DEPRECATION_WARNINGS)

# 提醒一定要在【import ragas 之前】设，否则 ragas 在 import 时抛的
# DeprecationWarning 已经打印出去了（旧版设得太晚，等于没生效）。
if HIDE_DEPRECATION_WARNINGS:
    warnings.filterwarnings("ignore", category=DeprecationWarning)

# 指标名 → 是否需要标准答案
NEEDS_REF = {"context_recall", "context_precision"}

# ragas 各版本返回的列名不完全一致，这里做兼容
COLUMN_ALIASES = {
    "faithfulness": ["faithfulness"],
    "answer_relevancy": ["answer_relevancy", "response_relevancy"],
    "context_recall": ["context_recall", "llm_context_recall"],
    "context_precision": ["llm_context_precision_with_reference",
                          "context_precision", "llm_context_precision"],
}


def pick_column(df, key):
    for c in COLUMN_ALIASES.get(key, [key]):
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------
#  评委 / 向量模型：全部走 Ollama 的 OpenAI 兼容接口（/v1）
# ---------------------------------------------------------
def build_clients():
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit(
            "❌ 缺少 openai 包：\n"
            "   pip install openai -i https://pypi.tuna.tsinghua.edu.cn/simple")

    # Ollama 的 OpenAI 兼容端点要求带 api_key，随便填
    return OpenAI(
        base_url=f"{OLLAMA_URL}/v1",
        api_key="ollama",
        timeout=EVAL_TIMEOUT,
        max_retries=EVAL_MAX_RETRIES,
    )


def build_judge(client):
    from ragas.llms import llm_factory
    return llm_factory(JUDGE_MODEL, provider="openai", client=client)


def build_judge_embeddings(client):
    """
    answer_relevancy 需要一个「向量模型」，ragas 只要求两个方法：
        embed_query(text) -> list[float]
        embed_documents(list[text]) -> list[list[float]]

    这里直接用 openai 客户端实现这两个方法（走 Ollama 的 /v1/embeddings）。
    ⚠️ 不要用 ragas 的 embedding_factory 现代接口：它返回的 OpenAIEmbeddings
       没有 embed_query，answer_relevancy 会抛
       AttributeError: 'OpenAIEmbeddings' object has no attribute 'embed_query'
       （实测踩过）；而自动降级到它的 legacy 接口会再套一层包装类。
    """

    class _OllamaEmbeddings:
        def __init__(self, client, model):
            self._client = client
            self._model = model
            self.run_config = None

        def set_run_config(self, run_config):      # ragas 可能调用
            self.run_config = run_config

        def embed_documents(self, texts):
            texts = [str(t) for t in texts]
            if not texts:
                return []
            r = self._client.embeddings.create(model=self._model, input=texts)
            return [list(d.embedding) for d in r.data]

        def embed_query(self, text):
            return self.embed_documents([text])[0]

    return _OllamaEmbeddings(client, EMBED_MODEL)


def build_run_config():
    """把 config.py 里的 EVAL_TIMEOUT / EVAL_MAX_WORKERS / EVAL_MAX_RETRIES 真正传进去"""
    try:
        from ragas import RunConfig
    except ImportError:                       # 兼容旧路径
        from ragas.run_config import RunConfig

    wanted = {"timeout": EVAL_TIMEOUT,
              "max_workers": EVAL_MAX_WORKERS,
              "max_retries": EVAL_MAX_RETRIES}
    try:
        allowed = {f.name for f in dataclasses.fields(RunConfig)}
    except TypeError:                         # 不是 dataclass
        allowed = set(wanted)
    kwargs = {k: v for k, v in wanted.items() if k in allowed}
    return RunConfig(**kwargs)


def main():
    if not DATA_OUT.exists():
        print(f"❌ 找不到数据集 {DATA_OUT}，请先运行 python run_batch.py")
        return

    try:
        from ragas import EvaluationDataset, evaluate
        from ragas.metrics import (Faithfulness, ResponseRelevancy,
                                   LLMContextRecall,
                                   LLMContextPrecisionWithReference)
    except ImportError as e:
        print(f"❌ ragas 没装好：{type(e).__name__}: {e}")
        print("   pip install ragas==0.4.3 openai "
              "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        return

    if HIDE_DEPRECATION_WARNINGS:
        warnings.filterwarnings("ignore", category=DeprecationWarning)

    df = pd.read_excel(DATA_OUT)
    print(f"✅ 读取 {len(df)} 条，列名={list(df.columns)}")

    required = ["user_input", "retrieved_contexts", "response", "reference"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"缺少列：{missing}\n当前列名：{list(df.columns)}")

    df["retrieved_contexts"] = df["retrieved_contexts"].fillna("").apply(
        lambda x: [c.strip() for c in str(x).split("|||") if c.strip()])
    df["reference"] = df["reference"].fillna("").apply(
        lambda x: (str(x).strip() or None) if str(x).strip().lower() != "nan" else None)

    empty_ctx = (df["retrieved_contexts"].apply(len) == 0).sum()
    if empty_ctx:
        print(f"⚠️ 有 {empty_ctx} 条没检索到上下文，这些样本分数不可信，请检查检索")

    # ---------- 决定跑哪些指标 ----------
    n_ref = df["reference"].notna().sum()
    has_ref = n_ref == len(df)

    if METRICS:                      # 手动指定
        wanted = list(METRICS)
        print(f"📊 按 config.py 指定：{wanted}")
    elif has_ref:                    # 自动：reference 齐全就全跑
        wanted = ["faithfulness", "answer_relevancy",
                  "context_recall", "context_precision"]
        print("📊 reference 齐全 → 四个指标全跑")
    else:
        wanted = ["faithfulness", "answer_relevancy"]
        print(f"📊 reference 只填了 {n_ref}/{len(df)} 条 → 只跑生成侧两个指标")

    # 需要 reference 但没填的，剔除并明确告知
    dropped = [m for m in wanted if m in NEEDS_REF and not has_ref]
    if dropped:
        print(f"   ⚠️ {dropped} 需要标准答案，已跳过")
        print(f"   👉 想评检索质量，先跑：python make_reference.py")
        wanted = [m for m in wanted if m not in dropped]

    if not wanted:
        print("❌ 没有可跑的指标")
        return

    # ---------- 幻觉探针题的特殊处理 ----------
    # 「文档未提及」的题，语义上不存在"该检索到什么"，
    # 硬算 context_recall 会得到 0（假警报），默认从检索指标里剔除。
    no_ans_mask = df["reference"].astype(str).str.contains("未提及")
    n_no_ans = int(no_ans_mask.sum())

    eval_df = df
    if n_no_ans and any(m in NEEDS_REF for m in wanted) \
            and EXCLUDE_NO_ANSWER_FROM_RETRIEVAL:
        print(f"\nℹ️ 检测到 {n_no_ans} 条「文档未提及」的幻觉探针题")
        print(f"   这类题在 context_recall/precision 上必然得 0（不存在可召回的内容）")
        print(f"   → 已从检索指标中剔除，但仍参与 faithfulness 评估（这才是它们的作用）")
        eval_df = df[~no_ans_mask].copy()

    dataset = EvaluationDataset.from_list(eval_df.to_dict(orient="records"))

    # ---------- 评委 ----------
    if JUDGE_MODEL == GEN_MODEL:
        print(f"⚠️ JUDGE_MODEL == GEN_MODEL == {JUDGE_MODEL}")
        print("   这等于让选手给自己打分，分数会虚高。建议换一个更大的评委模型。\n")

    client = build_clients()
    judge_llm = build_judge(client)
    judge_emb = build_judge_embeddings(client)
    run_config = build_run_config()

    factory = {
        "faithfulness":      lambda: Faithfulness(llm=judge_llm),
        "answer_relevancy":  lambda: ResponseRelevancy(llm=judge_llm,
                                                       embeddings=judge_emb),
        "context_recall":    lambda: LLMContextRecall(llm=judge_llm),
        "context_precision": lambda: LLMContextPrecisionWithReference(llm=judge_llm),
    }
    metrics = [factory[m]() for m in wanted if m in factory]

    print(f"\n⏳ 评委 = {JUDGE_MODEL}（Ollama /v1 · 并发 {EVAL_MAX_WORKERS} · "
          f"超时 {EVAL_TIMEOUT}s · 重试 {EVAL_MAX_RETRIES}）")
    print(f"   开始打分，共 {len(eval_df)} 条 × {len(metrics)} 个指标...\n")

    try:
        result = evaluate(dataset=dataset, metrics=metrics,
                          run_config=run_config)
    except Exception as e:                    # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        print(f"\n❌ 打分失败：{msg}")
        if "429" in msg or "concurrent" in msg.lower():
            print(f"\n   这是限流（云端模型常见）：")
            print(f"   · config.py 里 EVAL_MAX_WORKERS 已经是 {EVAL_MAX_WORKERS}，"
                  f"保持 1")
            print(f"   · 等 {EVAL_RETRY_WAIT}s 再跑一次")
        elif "connect" in msg.lower() or "refused" in msg.lower():
            print(f"\n   Ollama 连不上？先确认：curl {OLLAMA_URL}/api/tags")
        elif "does not support" in msg.lower() or "tools" in msg.lower():
            print(f"\n   评委模型不支持工具调用/结构化输出，ragas 的可靠性评分依赖它。")
            print(f"   换一个支持 tools 的模型（ollama list 里 capabilities 含 tools）")
        return

    scores = result.to_pandas()

    # ---------- 逐条得分 ----------
    show = ["user_input"] + [c for c in
                             (pick_column(scores, m) for m in wanted) if c]
    print("\n===== 逐条得分 =====")
    print(scores[show].to_string())

    # ---------- 均值 ----------
    print("\n===== 均值 =====")
    means = {}
    for m in wanted:
        c = pick_column(scores, m)
        if c and pd.api.types.is_numeric_dtype(scores[c]):
            means[m] = round(float(scores[c].mean()), 4)
    for k, v in means.items():
        print(f"  {k:20s} {v}")

    # ---------- NaN 比例 ----------
    print("\n===== NaN 比例（>5% 说明评委崩了）=====")
    for m in wanted:
        c = pick_column(scores, m)
        if c:
            print(f"  {m:20s} {scores[c].isna().mean():.2%}")

    # ---------- 诊断结论 ----------
    print("\n" + "═" * 60)
    print("【诊断】")
    fa = means.get("faithfulness")
    ar = means.get("answer_relevancy")
    cr = means.get("context_recall")
    cp = means.get("context_precision")

    def lvl(v, good=0.8, bad=0.5):
        if v is None:
            return "未测"
        if v >= good:
            return "🟢 良好"
        if v >= bad:
            return "🟡 一般"
        return "🔴 偏差"

    print(f"  生成侧  faithfulness      {lvl(fa)}   ({fa})")
    print(f"          answer_relevancy  {lvl(ar)}   ({ar})")
    if cr is not None or cp is not None:
        print(f"  检索侧  context_recall    {lvl(cr)}   ({cr})")
        print(f"          context_precision {lvl(cp)}   ({cp})")

    advice = []
    if cr is not None and cr < 0.7:
        advice.append("  · context_recall 低 → 该检的没检到：加大 TOP_K 或减小 CHUNK_SIZE")
    if cp is not None and cp < 0.7:
        advice.append("  · context_precision 低 → 检到了但排后面：开 rerank 或换更强 embedding")
    if cr is not None and cp is not None and cr >= 0.8 and cp >= 0.8 and fa is not None and fa < 0.8:
        advice.append("  · 检索没问题但 faithfulness 低 → 是生成器在编：换更大模型或收紧 prompt")
    if fa is not None and fa >= 0.95:
        advice.append("  · faithfulness 已饱和 → 加「文档里没有答案」的题，否则该指标失去区分力")
    if ar is not None and ar < 0.8:
        advice.append("  · answer_relevancy 低 → 答非所问：看检索相似度是否偏低，或回答过短")

    if advice:
        print("\n【建议】")
        print("\n".join(advice))
    else:
        print("\n  🟢 各项指标均在合理区间")

    DATA_DIR.mkdir(exist_ok=True)
    scores.to_csv(SCORE_OUT, index=False, encoding="utf-8-sig")
    print(f"\n💾 {VERSION} 结果已保存 → {SCORE_OUT}")
    print("👉 改参数后：build_index.py 会自动把 VERSION +1，然后重跑")
    print("   run_batch.py + run_eval.py，最后 python compare.py 对比各版本")


if __name__ == "__main__":
    main()
