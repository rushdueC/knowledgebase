# -*- coding: utf-8 -*-
"""
🎛️ 前端参数状态

设计：
    · 会话内参数放 st.session_state["params"]，问答页每次都读它 →
      前端改完立刻生效，不用重启 Streamlit
    · 点「同步到命令行」后写 data/ui_overrides.json，CLI 脚本也会读这套参数
    · 「恢复默认」回到 config.py 里的原始值（config.CONFIG_DEFAULTS）
"""
import streamlit as st

import config

# 前端可改的参数（顺序即界面顺序）
#   kind: int / float / bool / str / select
EDITABLE = [
    dict(key="TOP_K", kind="int", min=1, max=20,
         label="TOP_K", help="最终喂给模型的片段数（rerank 之后保留几段）"),
    dict(key="RETRIEVE_K", kind="int", min=4, max=60,
         label="RETRIEVE_K", help="召回候选池大小，必须 >= TOP_K"),
    dict(key="RERANK_ENABLED", kind="bool",
         label="启用 rerank", help="关掉就走「召回 → 直接取 TOP_K」，可用来做对照组"),
    dict(key="HYBRID_ENABLED", kind="bool",
         label="混合检索（BM25+向量）",
         help="向量召回 + BM25 关键词召回，用 RRF 融合；关掉 = 纯向量"),
    dict(key="GEN_MODEL", kind="str",
         label="GEN_MODEL", help="生成回答的模型（Ollama 里的名字）"),
    dict(key="TEMPERATURE", kind="float", min=0.0, max=1.0, step=0.1,
         label="TEMPERATURE", help="评测建议 0，要确定性"),
    dict(key="JUDGE_MODEL", kind="str",
         label="JUDGE_MODEL", help="评委模型，必须 != GEN_MODEL"),
]

# 只读展示（改了要重建库 / 重跑，不适合热改）
READONLY = [
    ("EMBED_MODEL", "嵌入模型"),
    ("CHUNK_SIZE", "chunk 大小"),
    ("CHUNK_OVERLAP", "chunk 重叠"),
    ("COLLECTION", "向量库"),
    ("NUM_CTX", "num_ctx"),
    ("HYBRID_RRF_K", "RRF 常数"),
    ("HYBRID_VEC_WEIGHT", "向量权重"),
    ("HYBRID_BM25_WEIGHT", "BM25 权重"),
    ("BM25_K1", "BM25 k1"),
    ("BM25_B", "BM25 b"),
]


def init():
    if "params" not in st.session_state:
        st.session_state["params"] = {
            d["key"]: getattr(config, d["key"], None) for d in EDITABLE
        }


def current() -> dict:
    init()
    return st.session_state["params"]


def get(key, default=None):
    """取当前生效值（前端覆盖优先）"""
    init()
    if key in st.session_state["params"]:
        return st.session_state["params"][key]
    return getattr(config, key, default)


def set_values(values: dict):
    init()
    st.session_state["params"].update(
        {k: v for k, v in values.items() if v is not None})


def reset_to_config():
    """回到 config.py 的原始值（忽略前端覆盖文件）"""
    st.session_state["params"] = {
        d["key"]: config.CONFIG_DEFAULTS.get(d["key"], getattr(config, d["key"], None))
        for d in EDITABLE
    }


def snapshot() -> dict:
    """给实验引擎用的完整参数快照（可编辑项 + 只读项）"""
    import metrics_store
    out = {}
    for key in metrics_store.PARAM_KEYS:
        if key in st.session_state.get("params", {}):
            out[key] = st.session_state["params"][key]
        elif hasattr(config, key):
            out[key] = getattr(config, key)
    return out


def is_overridden(key) -> bool:
    return key in getattr(config, "UI_OVERRIDES_APPLIED", {})


def overrides_file_exists() -> bool:
    from experiment import OVERRIDE_FILE
    return OVERRIDE_FILE.exists()
