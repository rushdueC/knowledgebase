# -*- coding: utf-8 -*-
"""
🔎 检索核心 —— 被 Streamlit 界面、实验引擎、CLI 脚本共用

为什么要单独抽出来：
    同一套「向量粗排 → rerank 精排」原来在三处各写了一遍
    （app.py / run_batch.py / make_reference.py），参数一改就要改三处，
    前端调参和命令行调参的行为很容易不一致。
    现在统一到这里，参数由调用方显式传入。

设计要点：
    · 不 import streamlit，纯 Python，CLI 和 Streamlit 都能用
    · 索引只加载一次（实例内缓存），前端每次提问不会重读 280KB 的 json
    · 同时返回「排序分数」和「余弦相似度」两个 top1：
      cross-encoder / RRF 分数和余弦不是一个量纲，门禁判断只能用后者
    · 支持「混合检索」：向量召回 + BM25 召回，用 RRF 融合名次
      （BM25 在 bm25.py，融合参数见 config.py 的 HYBRID_* / BM25_*）
"""
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests

from bm25 import BM25

BASE = Path(__file__).resolve().parent


# ---------------------------------------------------------------- 配置读取
def cfg(name, default=None):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


@dataclass
class SearchResult:
    docs: list = field(default_factory=list)
    metas: list = field(default_factory=list)
    scores: list = field(default_factory=list)      # 实际排序用的分数
    vec_top1: float | None = None                   # 余弦相似度 top1（永远可比）
    note: str = "关闭"                              # rerank 状态说明
    elapsed_ms: float = 0.0
    reranked: bool = False
    hybrid: bool = False                            # 本次是否走了 BM25+向量 融合


def rrf_fuse(rankings, weights=None, rrf_k=60.0, k=20):
    """
    倒数排名融合（Reciprocal Rank Fusion）。

    为什么不用「分数加权求和」：BM25 分数没上界、余弦在 0~1、
    cross-encoder 又是另一个分布，直接相加等于让量纲决定结果。
    RRF 只看名次：score(d) = Σ w_i / (k + rank_i)，量纲无关、零调参、
    在实践中几乎总是优于分数归一化。

    rankings：[[idx, idx, ...], ...]，每个列表是某一路检索的降序结果
    weights ：每一路的权重（None = 全 1）
    rrf_k   ：平滑常数，60 是原论文的经验值
    k       ：返回多少个
    """
    weights = list(weights or [1.0] * len(rankings))
    agg = {}
    for ranked, w in zip(rankings, weights):
        if not w:
            continue
        for rank, idx in enumerate(ranked):
            agg[idx] = agg.get(idx, 0.0) + float(w) / (rrf_k + rank + 1)
    # 分数降序；同分按索引升序，保证结果稳定可复现
    order = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))
    return order[:k]


# ---------------------------------------------------------------- embedding
def embed_one(text, embed_model=None, ollama_url=None):
    """把一句话变成单位向量（兼容新旧两个 Ollama 端点）

    keep_alive=60s：嵌入模型用完 1 分钟就卸载。
    否则它会和生成模型（qwen3:8b 占 6GB）同时驻留 —— 2026-11-06 实测，
    bge-m3 + qwen3:8b + rerank 三模型并存时可用内存只剩 2GB，单题生成从
    4 分钟涨到 9.5 分钟，最后一题直接 900 秒超时。
    """
    embed_model = embed_model or cfg("EMBED_MODEL", "qwen3-embedding:0.6b")
    ollama_url = ollama_url or cfg("OLLAMA_URL", "http://localhost:11434")
    keep = cfg("EMBED_KEEP_ALIVE", "60s")

    try:
        r = requests.post(f"{ollama_url}/api/embed",
                          json={"model": embed_model, "input": [text],
                                "keep_alive": keep},
                          timeout=300)
        if r.status_code == 200:
            v = np.array(r.json()["embeddings"][0], dtype="float32")
        else:
            raise RuntimeError(r.text[:200])
    except Exception:
        r = requests.post(f"{ollama_url}/api/embeddings",
                          json={"model": embed_model, "prompt": text,
                                "keep_alive": keep},
                          timeout=300)
        r.raise_for_status()
        v = np.array(r.json()["embedding"], dtype="float32")
    n = np.linalg.norm(v)
    return v / n if n else v


# ---------------------------------------------------------------- 索引加载
def load_index(vec_path=None, idx_path=None):
    """返回 (arr, chunks, metas)；库不存在时抛 RuntimeError（带修复提示）"""
    if vec_path is None or idx_path is None:
        db_dir = Path(cfg("DB_DIR", BASE / "vector_store"))
        collection = cfg("COLLECTION", "")
        vec_path = vec_path or db_dir / f"{collection}.npy"
        idx_path = idx_path or db_dir / f"{collection}.json"

    vec_path, idx_path = Path(vec_path), Path(idx_path)
    if not vec_path.exists() or not idx_path.exists():
        raise RuntimeError(
            f"找不到向量库 {vec_path.name}\n"
            f"  期望位置：{vec_path.parent}\n"
            f"  👉 请先运行：python build_index.py")

    arr = np.load(vec_path)
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    return arr, idx["chunks"], idx.get("metas", [])


# ---------------------------------------------------------------- 文档文件夹
_DIR_MAP_CACHE = None


def _legacy_source_dir_map():
    """
    旧索引的 metas 里没有 doc_dir 时的兜底：按文件名去 docs/ 反查所在文件夹。

    只用于兼容旧库；重跑一次 build_index.py 之后就不需要它了。
    同名文件出现在多个文件夹时记为 "?"（无法判定，不参与筛选）。
    """
    global _DIR_MAP_CACHE
    if _DIR_MAP_CACHE is not None:
        return _DIR_MAP_CACHE
    docs = Path(cfg("DOCS_DIR", BASE / "docs"))
    found = {}
    if docs.exists():
        for p in docs.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(docs).as_posix()
            dir_of = rel.rsplit("/", 1)[0] if "/" in rel else ""
            found.setdefault(p.name, set()).add(dir_of)
    _DIR_MAP_CACHE = {k: (v.pop() if len(v) == 1 else "?") for k, v in found.items()}
    return _DIR_MAP_CACHE


def meta_dir(meta, legacy=None):
    """取一个 chunk 的所属文件夹；根目录返回 ""，判定不了返回 "?" """
    d = meta.get("doc_dir") if isinstance(meta, dict) else None
    if d is not None:
        return str(d)
    legacy = _legacy_source_dir_map() if legacy is None else legacy
    return str(legacy.get((meta or {}).get("source"), "?"))


def doc_dir_stats(index=None):
    """
    统计每个文件夹有多少块 / 多少份文档，给前端筛选器用。

    返回 {"需求文档": {"chunks": 29, "docs": 11}, ...}（含根目录 "" 与 "?"）
    """
    _, _, metas = index if index is not None else load_index()
    legacy = _legacy_source_dir_map()
    stats = {}
    for m in metas:
        d = meta_dir(m, legacy)
        item = stats.setdefault(d, {"chunks": 0, "sources": set()})
        item["chunks"] += 1
        item["sources"].add((m or {}).get("source", "?"))
    return {d: {"chunks": v["chunks"], "docs": len(v["sources"])}
            for d, v in stats.items()}


# ---------------------------------------------------------------- 检索器
class Retriever:
    """
    一次配置、多次提问。

    参数全部显式传入（默认值来自 config.py），这样前端把用户改的值
    直接塞进来就能生效，不用去改全局配置。

    doc_dirs：只在这些 docs 子文件夹里检索（空/None = 全部）。
    use_hybrid：是否启用「BM25 + 向量」混合检索（None = 读 config）。
    """

    def __init__(self, *, top_k=None, pool_k=None, use_rerank=None,
                 reranker=None, vec_path=None, idx_path=None,
                 embed_model=None, ollama_url=None, index=None,
                 doc_dirs=None, use_hybrid=None,
                 rrf_k=None, vec_weight=None, bm25_weight=None,
                 bm25_k1=None, bm25_b=None):
        self.top_k = int(top_k if top_k is not None else cfg("TOP_K", 5))
        self.pool_k_default = int(pool_k if pool_k is not None
                                  else cfg("RETRIEVE_K", 20))
        self.embed_model = embed_model or cfg("EMBED_MODEL",
                                              "qwen3-embedding:0.6b")
        self.ollama_url = ollama_url or cfg("OLLAMA_URL",
                                            "http://localhost:11434")
        self._index = index
        self._vec_path = vec_path
        self._idx_path = idx_path
        # 空列表 = 不筛选（全部文档）
        self.doc_dirs = [str(d) for d in (doc_dirs or []) if str(d) != ""] \
            if doc_dirs else []

        # 混合检索：显式开关优先，其次看 config
        if use_hybrid is None:
            use_hybrid = bool(cfg("HYBRID_ENABLED", False))
        self.use_hybrid = bool(use_hybrid)
        self.rrf_k = float(rrf_k if rrf_k is not None
                           else cfg("HYBRID_RRF_K", 60) or 60)
        self.vec_weight = float(vec_weight if vec_weight is not None
                                else cfg("HYBRID_VEC_WEIGHT", 1.0) or 1.0)
        self.bm25_weight = float(bm25_weight if bm25_weight is not None
                                 else cfg("HYBRID_BM25_WEIGHT", 1.0) or 1.0)
        self.bm25_k1 = float(bm25_k1 if bm25_k1 is not None
                             else cfg("BM25_K1", 1.5) or 1.5)
        self.bm25_b = float(bm25_b if bm25_b is not None
                            else cfg("BM25_B", 0.75) or 0.75)
        self._bm25 = None

        # rerank：显式开关优先，其次看 config
        from rerank import Reranker
        if use_rerank is None:
            use_rerank = bool(cfg("RERANK_ENABLED", False))
        self.use_rerank = bool(use_rerank)
        if reranker is not None:
            self.reranker = reranker
        elif self.use_rerank:
            self.reranker = Reranker()
        else:
            self.reranker = None

    # ---- 文档范围过滤 ----
    def _doc_mask(self, metas):
        """返回布尔掩码（None = 不筛选）"""
        if not self.doc_dirs:
            return None
        want = set(self.doc_dirs)
        legacy = _legacy_source_dir_map()
        mask = np.zeros(len(metas), dtype=bool)
        for i, m in enumerate(metas):
            mask[i] = meta_dir(m, legacy) in want
        return mask

    def _apply_mask(self, sims, mask):
        if mask is None:
            return sims, sims.shape[0]
        avail = int(mask.sum())
        if avail == 0:
            raise RuntimeError(
                "选中的文档范围里没有内容："
                + "、".join(self.doc_dirs)
                + "\n请换一个文件夹，或点「全部」取消筛选。")
        return np.where(mask, sims, -np.inf), avail

    # ---- 索引（懒加载，实例内复用）----
    def index(self):
        if self._index is None:
            self._index = load_index(self._vec_path, self._idx_path)
        return self._index

    def invalidate_index(self):
        self._index = None
        self._bm25 = None

    def _bm25_index(self, chunks):
        """BM25 索引懒构建（165 块约几十毫秒，只建一次）"""
        if self._bm25 is None:
            self._bm25 = BM25(chunks, k1=self.bm25_k1, b=self.bm25_b)
        return self._bm25

    def _allowed_list(self, metas):
        """给 BM25 用的布尔白名单（None = 不过滤）"""
        mask = self._doc_mask(metas)
        return None if mask is None else mask.tolist()

    # ---- 只要向量召回（不 rerank、不融合）----
    def vector_search(self, question, k=None):
        """纯向量检索，返回 (docs, metas, sims)。给「先粗排再精排」的第一步用"""
        arr, chunks, metas = self.index()
        k = min(int(k or self.pool_k_default), len(chunks))
        if k <= 0:
            raise RuntimeError("向量库是空的，请先 python build_index.py")
        qv = embed_one(question, self.embed_model, self.ollama_url)
        sims = arr @ qv
        sims, avail = self._apply_mask(sims, self._doc_mask(metas))
        k = min(k, avail)
        top = np.argsort(-sims)[:k]
        return ([chunks[i] for i in top],
                [metas[i] if i < len(metas) else {} for i in top],
                [float(sims[i]) for i in top])

    # ---- 主入口 ----
    def search(self, question, top_k=None, pool_k=None) -> SearchResult:
        """
        检索：向量 →（可选）BM25 融合 →（可选）rerank → 取 top_k

        · use_hybrid=True 时，向量和 BM25 各取 pool 个，用 RRF 合并成一个候选池
        · 之后如果开了 rerank，cross-encoder 会在这个**合并后的池子**里精排
        """
        t0 = time.perf_counter()
        arr, chunks, metas = self.index()

        top_k = int(top_k or self.top_k)
        # 开 rerank 或混合检索时，先粗排出一批候选，否则直接取 top_k
        need_pool = self.reranker is not None or self.use_hybrid
        base = max(int(pool_k or self.pool_k_default), top_k) if need_pool \
            else top_k
        base = min(base, len(chunks))
        if base <= 0:
            raise RuntimeError("向量库是空的，请先 python build_index.py")

        qv = embed_one(question, self.embed_model, self.ollama_url)
        sims = arr @ qv

        # 文档范围筛选：把范围外的相似度打成 -inf，排序时自然沉底
        sims, avail = self._apply_mask(sims, self._doc_mask(metas))
        pool = min(base, avail)

        vec_ranked = [int(i) for i in np.argsort(-sims)[:pool]]
        # 允许范围内的最大余弦：门禁/诊断只认这个（和 rank 用什么无关）
        vec_top1 = float(np.max(sims)) if avail else None

        hybrid_used = False
        order, rank_scores = vec_ranked, [float(sims[i]) for i in vec_ranked]

        if self.use_hybrid:
            bm_hits = self._bm25_index(chunks).search(
                question, pool, allowed=self._allowed_list(metas))
            bm_ranked = [i for i, _ in bm_hits]
            if bm_ranked:
                fused = rrf_fuse([vec_ranked, bm_ranked],
                                 weights=[self.vec_weight, self.bm25_weight],
                                 rrf_k=self.rrf_k, k=pool)
                order = [i for i, _ in fused]
                rank_scores = [float(s) for _, s in fused]
                hybrid_used = True

        cand_docs = [chunks[i] for i in order]
        cand_metas = [metas[i] if i < len(metas) else {} for i in order]

        note = "关闭"
        if self.reranker is not None:
            # rank() 返回 (原始索引, 分数)，用它才能把 metas 正确对上
            scored = self.reranker.rank(question, cand_docs, min(top_k,
                                                                len(cand_docs)))
            if scored is not None:
                note = f"✅ {self.reranker.backend}" if not self.reranker.failed \
                    else "⚠️ 降级"
                return SearchResult(
                    docs=[cand_docs[i] for i, _ in scored],
                    metas=[cand_metas[i] for i, _ in scored],
                    scores=[float(s) for _, s in scored],
                    vec_top1=vec_top1, note=note,
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                    reranked=True, hybrid=hybrid_used)
            reason = (self.reranker.fail_reason or "未启用").splitlines()[0]
            note = f"⚠️ 降级（{reason[:36]}）"

        return SearchResult(
            docs=cand_docs[:top_k],
            metas=cand_metas[:top_k],
            scores=rank_scores[:top_k],
            vec_top1=vec_top1, note=note,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            reranked=False, hybrid=hybrid_used)
