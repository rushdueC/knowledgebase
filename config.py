# -*- coding: utf-8 -*-
"""
=========================================================
  所有可调参数集中在这里 —— 调参只改这一个文件
=========================================================
改完参数后，按这个顺序重跑：
    python predownload_rerank.py  # 只在第一次用 local rerank 时需要
    python build_index.py         # 改了 CHUNK_* / EMBED_MODEL 才需要重跑
    python run_batch.py           # 每次都跑
    python make_reference.py      # 需要检索侧两个指标时
    python run_eval.py            # 每次都跑
=========================================================
"""
import sys as _sys
from pathlib import Path

# ---------- Windows 控制台编码兜底 ----------
# Windows 的 Python 在 stdout 被重定向（`> log.txt`、管道、某些 IDE）时
# 会退回 GBK 编码，脚本里的 ✅ 📄 🔢 会直接抛 UnicodeEncodeError 把脚本打断。
# 这里统一改成 UTF-8（config 被所有脚本 import，所以放这里一处即可）。
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

BASE = Path(__file__).resolve().parent

# ---------- 路径 ----------
DOCS_DIR   = BASE / "docs"        # 原始文档放这里（txt / md / pdf / docx）
DB_DIR     = BASE / "vector_store"  # 向量库（自动生成，纯 numpy，无需 chromadb）
DATA_DIR   = BASE / "data"        # 数据集 + 结果
QUESTIONS  = BASE / "questions.txt"

# ---------- Ollama ----------
OLLAMA_URL   = "http://localhost:11434"

# 嵌入模型：v4 起换成 bge-m3
# 依据（2026-11-06，398 块 / 20 题 / 同题同参考答案）：
#   qwen3-embedding:0.6b → recall 0.9271 / precision 0.9003，建库 15.8 分钟
#   bge-m3               → recall 0.9479 / precision 0.9097，建库 11.9 分钟
#   11 组对照里唯一"两个指标都不亏"的改动（recall 是唯一不退化的召回改善）
# ⚠️ 换它等于换库：COLLECTION 会变成 kb_bge-m3_800，必须重建索引（约 12 分钟）
EMBED_MODEL  = "bge-m3"

# 被评测的生成模型
GEN_MODEL    = "qwen3:8b"

# 评委模型 —— 必须不同于 GEN_MODEL，否则等于选手自己打分，分数会虚高。
# 默认用云端大模型：比本地 8b 可信得多，而且不占本机算力（本地 CPU 跑评测是小时级）。
# 想完全离线就换成已装的本地模型，例如 "qwen3:8b"（此时 GEN_MODEL 要换成 qwen3:4b）。
JUDGE_MODEL  = "gemma4:31b-cloud"

# ---------- 切块参数 ----------
CHUNK_SIZE    = 800      # 每块字符数（中文 500~1200 都合理）
CHUNK_OVERLAP = 100      # 块间重叠（建议 chunk 的 10%~20%）

# ---------- 检索参数 ----------
# 最终喂给模型几段（rerank 后保留的数量）
#
# 📊 2026-11-02 实测（同一批 8 题，快速检索实验，rerank 开）：
#        TOP_K=5 → context_recall 1.000 / context_precision 0.903
#        TOP_K=3 → context_recall 0.875 / context_precision 0.927
#    precision 那 +0.024 落在评委噪声内（同配置重复跑就差 0.0245），
#    recall 却是实打实掉了 0.125（「市场洞察有哪些模块」直接 1.0 → 0.0）。
#    ⚠️ 想回到更稳的配置：把下面这行改成 TOP_K = 5。
#    （顺便一提：改 TOP_K 几乎不省时间，rerank 的耗时在候选池 RETRIEVE_K 上；
#      真想提速就把 RETRIEVE_K 从 20 降到 10~12。）
# 2026-11-05：已按要求改回 5，并在新语料上重跑。
TOP_K = 5

# 检索超时秒数（Streamlit 界面用）
# Ollama 卡住时（模型未加载 / 服务假死）界面不再无限转圈
#
# ⚠️ 开了 rerank 之后别再设 60：实测单次检索（20 个候选逐个过 cross-encoder）
#    在纯 CPU 上约 45 秒，机器一忙就会误报「检索超过 60s 未完成」。
#    180 秒能覆盖冷启动 + rerank 的正常耗时，又不会真的无限等。
SEARCH_TIMEOUT = 180

# 生成超时秒数（界面里「等首字」的耐心）
# ⚠️ 必须给足：本地 8B 在 CPU 上「加载模型 + 预填充 2400 字上下文」可能就要
#    2~4 分钟，180 秒会误判成失败。云端模型几秒就出首字，这个值不影响它。
GEN_TIMEOUT = 600

# ---------- 重排（rerank）----------
# 用来提升 context_precision：向量检索排序不准，rerank 能重排顺序。
# 只改顺序、不改变召回集合，所以 recall 基本不变、precision 会涨。
#
# 三种状态：
#   关闭               → 纯向量检索（最省事，先跑通流程就设 False）
#   True + "local"     → 本地 cross-encoder，不用 Docker（推荐）
#   True + "http"      → 调独立服务（需 Docker 起 FastGPT rerank 镜像）
#
# ⭐ 用 local 后端之前，先跑一次预下载，否则运行时会联网下载（可能卡死）：
#       python predownload_rerank.py
RERANK_ENABLED = True
RERANK_BACKEND = "local"

# http 后端用：FastGPT 官方 rerank 镜像，docker-compose 里加一段就能起：
#   reranker:
#     image: registry.cn-hangzhou.aliyuncs.com/fastgpt/rerank:v0.2
#     ports: ["6006:6006"]
#     environment:
#       - MODEL_NAME=bge-reranker-base
RERANK_URL    = "http://localhost:6006/v1/rerank"
RERANK_AUTH   = ""                  # 仅 http 后端用；填服务端要求的 token，本地后端可留空
RERANK_TIMEOUT = 120

# ---- local / http 共用 ----
# 中文场景用 v2-m3（多语言，约 1.1GB，效果最好）；
# 磁盘/显存紧张就换 BAAI/bge-reranker-base（约 400MB）。
RERANK_MODEL  = "BAAI/bge-reranker-v2-m3"

# 本地模型目录（predownload_rerank.py 默认就下到这里）。
# 用 BASE 拼出来，换电脑不用改路径。
# 目录不存在时 rerank 会明确报错并降级，不会偷偷联网下载。
RERANK_LOCAL_PATH = str(BASE / "models" / "bge-reranker-v2-m3")

# local 后端推理时的 batch size；显存小就调小
RERANK_BATCH_SIZE = 16

# HuggingFace 镜像。国内直连 huggingface.co 基本必超时（WinError 10060）。
# 设为 "https://hf-mirror.com" 可走国内镜像；设空串则用官方源。
HF_ENDPOINT = "https://hf-mirror.com"

# 本地模型加载的超时秒数。
# ⚠️ 非常重要：sentence-transformers 下载模型时如果网络不通，
#    会无限重试且【无法被打断】。这里用线程超时强制掐断，
#    超时后自动降级，避免界面永远卡在「检索中...」。
RERANK_LOAD_TIMEOUT = 180

# 初召回数量：rerank 的候选池，必须 >= TOP_K
# 不开启 rerank 时这个值不生效（直接用 TOP_K）
RETRIEVE_K = 20

# ---------- 混合检索（BM25 + 向量）----------
# 为什么要混合：
#   向量检索擅长「语义相近」，但对专有名词/缩写/数字常失灵
#   （问「IRR 二分法」，向量可能召回一堆泛泛讲 IRR 的段落，漏掉写着「二分法」那段）；
#   BM25 只认字面命中，正好补这块。两路召回用 RRF 融合名次。
#
# 融合方式为什么用 RRF（倒数排名融合）而不是分数加权：
#   BM25 分数没有上界、余弦在 0~1、cross-encoder 又是另一个分布，
#   直接相加等于让「谁的量纲大」决定结果；RRF 只看名次：Σ w/(k+rank)。
#
# 📊 2026-11-04 实测（8 题、TOP_K=5、rerank 开，同题同评委）：
#       纯向量 → recall 1.000 / precision 0.903
#       混合   → recall 0.875 / precision 0.910
#    只有 1 题变了（「市场洞察有哪些模块」recall 1.0→0.0，precision 0.59→0.64），
#    其余 7 题逐题完全一致。→ 在这套 8 题上**看不出收益，还丢了 1 题召回**。
#    但关键词型查询（「IRR 二分法」「SPAN 23 象限」）混合检索明显更准，
#    只是那类题题库里还没有。所以：
#    ⚠️ 默认关。等题库扩到 20~30 题（含关键词型）再判定要不要默认开。
#       想现在就用：把这里改 True，或在控制台「参数与指标」页勾上。
HYBRID_ENABLED = False     # 关掉 = 纯向量检索；开 = 向量 + BM25 融合
HYBRID_RRF_K = 60          # RRF 平滑常数，60 是论文默认值，一般不用改
HYBRID_VEC_WEIGHT = 1.0    # 向量这一路的权重
HYBRID_BM25_WEIGHT = 1.0   # BM25 这一路的权重（想让关键词更强势就调大，如 1.5）

# BM25 自身的两个参数（Okapi BM25）
BM25_K1 = 1.5              # 词频饱和：越大越「词出现得多就占便宜」，1.2~2.0
BM25_B = 0.75              # 长度惩罚：0=不惩罚，1=完全按长度归一，0.75 常用

# ---------- 生成参数 ----------
TEMPERATURE = 0.0        # 评测必须 0，要确定性
NUM_CTX     = 16384      # Ollama 默认 2048 会静默截断，必须调大

# 关闭 qwen3 的「思考模式」。
# 实现方式：在 user 消息末尾加 Qwen3 官方的软开关 /no_think。
# （旧版这里拼的是一串来历不明的 <arg_key:...>，它不是 qwen3 的开关，
#   模型会把它当正文评论出来，等于既没关思考又污染了答案。）
DISABLE_THINK = True

# ---------- 评测指标（新增：四大指标全开）----------
# 四个指标分两类：
#   生成侧（不需要 reference）：faithfulness      —— 回答有没有编造
#                              answer_relevancy  —— 有没有答到点上
#   检索侧（需要 reference）  ：context_recall    —— 该检的是不是漏了
#                              context_precision —— 检到的是不是排在前面
#
# 留空 = 自动（reference 填满了就全跑，没填满只跑生成侧）
# 也可手动指定，例如 ["faithfulness", "context_recall"]
METRICS = []

# 生成标准答案草案用的模型（make_reference.py）
# ⚠️ 必须不同于 GEN_MODEL，否则等于让选手自己出标准答案
# 💡 留空 "" 则自动跟随 JUDGE_MODEL —— 这样你只改 JUDGE_MODEL 一处就够了
REF_MODEL = ""

# ⭐ 关键：起草 reference 时用多大的 top_k 来检索
# 默认 20，必须【远大于】正式评测的 TOP_K（默认 5）
#
# 为什么必须更大：
#   reference 若从「本次 TOP_K 召回的原文」起草，那它的每句话必然能在
#   这些原文里找到 → context_recall 恒等于 1.0，指标彻底失效
#   （实测 8 条全部 1.0，标准差 0.0000）
#
#   用更大的 k 起草，reference 的信息面就比正式召回的 TOP_K 更宽，
#   recall 才真正衡量「你用 top_k=5 覆盖了多少参考答案的要点」
REF_TOP_K = 20

# 相似度低于此值 → make_reference 直接判定「文档未提及」，不调模型起草
# （低于 0.45 说明检索到的根本不是相关内容，模型起草只会瞎编）
#
# ⚠️ 这个阈值只对【余弦相似度】有效，和开不开 rerank 无关：
#     cross-encoder 分数是另一个量纲（实测相关句 0.05、无关句 0.000），
#     拿 0.45 去卡它会把正常题全判成「文档未提及」。
#     所以 run_batch 会把两个分数都写进数据集：
#         top1_similarity         —— 实际排序用的分数（开 rerank 时是 cross-encoder 分数）
#         top1_vector_similarity  —— 永远是余弦，门禁/诊断一律用它
SIM_THRESHOLD = 0.45

# 「文档未提及」的幻觉探针题，是否从主评估中剔除
# True（推荐）：这类题会同时污染三个指标 ——
#              faithfulness 虚高（拒答没有 claim，算出满分）
#              relevancy   虚低（惩罚了正确的拒答行为）
#              recall      假警报（不存在「该召回什么」，必然得 0）
#              → 剔除后单独用「正确拒答率」衡量
# False：全量参与（分数会失真，但口径统一）
EXCLUDE_NO_ANSWER_FROM_RETRIEVAL = True

# 是否屏蔽 ragas 的 DeprecationWarning（屏幕干净些）
# 改 False 可以看到完整警告
HIDE_DEPRECATION_WARNINGS = True

# 打分失败时的重试次数（云端/本地模型偶发超时会产生 NaN）
# 实测默认 1 次时 NaN 率可达 25%，提高到 4 能显著改善。
# 生效位置：run_eval.py 建 OpenAI 客户端时的 max_retries
EVAL_MAX_RETRIES = 4
# 单次打分的超时秒数，云端模型建议放宽（OpenAI 客户端的 timeout）
EVAL_TIMEOUT = 300

# ⭐ 并发数 —— 云端模型报 429 时把这个调到 1
# 云端模型（如 gemma4:31b-cloud）通常有并发限制，ragas 默认并发较高，
# 会触发 "429 too many concurrent requests"。
# 本地 Ollama 也建议 1~2，本地模型扛不住高并发。
# 生效位置：run_eval.py 的 RunConfig(max_workers=...)
EVAL_MAX_WORKERS = 1

# 429 重试之间的等待秒数（云端的限流窗口通常是 60s）
# 生效位置：上面的 OpenAI 客户端（超时/重试）+ 429 时的提示文案
EVAL_RETRY_WAIT = 60

# ---------- 版本标记（调一次参数就 +1，方便多版本对比）----------
VERSION = "v4"
AUTO_BUMP_VERSION = True

# ---------- 前端覆盖（data/ui_overrides.json）----------
# Streamlit「参数与指标」页改的参数会写进这个文件，这样【前端】和
# 【命令行脚本】跑的是同一套参数，不会出现「界面显示 TOP_K=3，脚本还在用 5」。
# 想回到 config.py 里的值：删掉 data/ui_overrides.json，或点界面上的「恢复默认」。
# 只接受下面白名单里的键，且会做类型校验 —— 写坏了也不会让脚本崩溃。
import json as _json

_OVERRIDABLE = {
    "EMBED_MODEL": str, "GEN_MODEL": str, "JUDGE_MODEL": str, "REF_MODEL": str,
    "TOP_K": int, "RETRIEVE_K": int, "CHUNK_SIZE": int, "CHUNK_OVERLAP": int,
    "NUM_CTX": int, "REF_TOP_K": int, "EVAL_MAX_WORKERS": int,
    "TEMPERATURE": float, "SIM_THRESHOLD": float,
    "RERANK_ENABLED": bool, "DISABLE_THINK": bool,
    "RERANK_BACKEND": str, "RERANK_MODEL": str, "RERANK_LOCAL_PATH": str,
    "AUTO_BUMP_VERSION": bool,
    # 混合检索（BM25 + 向量）
    "HYBRID_ENABLED": bool, "HYBRID_RRF_K": int,
    "HYBRID_VEC_WEIGHT": float, "HYBRID_BM25_WEIGHT": float,
    "BM25_K1": float, "BM25_B": float,
}

_OVERRIDE_FILE = BASE / "data" / "ui_overrides.json"

# 记录 config.py 里的原始值（被前端覆盖前），供界面「恢复默认」用
CONFIG_DEFAULTS = {_k: globals().get(_k) for _k in _OVERRIDABLE}

UI_OVERRIDES_APPLIED = {}
if _OVERRIDE_FILE.exists():
    try:
        _ov = _json.loads(_OVERRIDE_FILE.read_text(encoding="utf-8"))
    except Exception as _e:                       # noqa: BLE001
        _ov = {}
        print(f"⚠️ 读取 {_OVERRIDE_FILE.name} 失败，已忽略：{_e}")
    for _k, _want in _OVERRIDABLE.items():
        if _k not in _ov:
            continue
        _v = _ov[_k]
        try:
            if _want is bool:
                _v = bool(_v) if isinstance(_v, (bool, int)) else \
                    str(_v).strip().lower() in ("1", "true", "yes", "on")
            elif _want is int:
                _v = int(_v)
            elif _want is float:
                _v = float(_v)
            else:
                _v = str(_v)
        except (TypeError, ValueError):
            print(f"⚠️ ui_overrides 里的 {_k}={_v!r} 类型不对，已忽略")
            continue
        globals()[_k] = _v
        UI_OVERRIDES_APPLIED[_k] = _v

# ---------- 自动推导（不要手动改）----------
def _safe(s):
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s)

# REF_MODEL 留空则自动跟随 JUDGE_MODEL，避免"改了评委忘了改起草模型"
if not REF_MODEL:
    REF_MODEL = JUDGE_MODEL

# collection 名带上嵌入模型+块大小：换模型/换块大小会自动建新库，避免维度冲突
COLLECTION = f"kb_{_safe(EMBED_MODEL)}_{CHUNK_SIZE}"

DATA_OUT  = DATA_DIR / f"eval_data_{VERSION}.xlsx"
SCORE_OUT = DATA_DIR / f"scores_{VERSION}.csv"

RAG_SYSTEM_PROMPT = """你是一个严格的问答助手，只根据提供的【参考资料】回答问题。

必须遵守：
1. 答案必须完全来自参考资料，禁止使用你自己的知识补充或推测。
2. 如果参考资料中没有答案，只回答：「根据提供的资料无法回答该问题。」不要编造。
3. 用参考资料原文中的表述，不要改写数字、名称、时间。
4. 用中文回答，简洁准确，不要重复问题。"""

# 生成标准答案（reference）用的提示词
# 关键：要求「摘录原文要点」，而不是自由发挥，
# 否则 ContextRecall 会把「reference 里的话在上下文找不到」判成检索失败
REFERENCE_PROMPT = """你是评测数据集的标注员。请为下面这个问题写出「标准答案」。

【规则 - 非常重要】
1. 标准答案必须完全来自【参考资料】，用参考资料中的原文措辞。
2. 只写答案要点，不要写解释、不要复述问题、不要加「根据资料」这类前缀。
3. 长度控制在 1~3 句话，涵盖参考资料中所有与该问题直接相关的要点。
4. 如果参考资料中确实没有该问题的答案，只输出四个字：文档未提及
5. 不要输出任何多余内容，直接给答案。

【参考资料】
{contexts}

【问题】
{question}

【标准答案】"""
