# -*- coding: utf-8 -*-
"""
🔀 重排模块（rerank）—— 提升 context_precision 的最有效手段

为什么需要它：
    向量检索靠「整体语义相似度」，粗排能力可以，但排序不准。
    实测第8条：最相关的片段排在第5位，完全无关的麦肯锡7S排在第3位
    → recall=1.0（找回来了）但 precision 只有 0.45（排得一塌糊涂）

    rerank 用 cross-encoder 逐个精算「问题 vs 片段」的相关性，
    专门治这个。它不改变召回集合，只改变顺序。

两种后端：
    local —— 本地跑 CrossEncoder（sentence-transformers），不用 Docker，推荐
    http  —— 调独立 rerank 服务（FastGPT 镜像 / Infinity / TEI / Xinference）
    其它值 —— 视为关闭

⚠️ 容错设计：rerank 失败时自动降级为纯向量检索，不会中断整个流程。
   但降级是「静默」的（返回 None 分数），调用方必须检查 None，
   不能只看有没有抛异常 —— 否则会误报「rerank 可用」。

⭐ 本版修掉的坑（旧版文档承诺了但代码里根本没实现）：
   1. RERANK_LOCAL_PATH 从来没被读过 → 填了本地目录也照样联网下载
   2. HF_ENDPOINT（国内镜像）从来没被设进环境变量 → 必走 huggingface.co 超时
   3. RERANK_LOAD_TIMEOUT 从来没被用过 → 模型下载/加载会无限卡死界面
   4. rerank() 只返回重排后的文档，不返回原始索引 → 调用方无法正确映射来源
      （run_batch 因此把 metas 映射错了位），现在提供 rank() 返回 (索引, 分数)
"""
import os
import threading
import requests
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"


def _silence_hf_progress():
    """关掉 transformers / huggingface_hub 的进度条与啰嗦日志（只影响控制台观感）"""
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:                                # noqa: BLE001
        pass
    try:
        import huggingface_hub
        from huggingface_hub.utils import logging as hub_logging
        hub_logging.set_verbosity_error()
        huggingface_hub.utils.disable_progress_bars()
    except Exception:                                # noqa: BLE001
        pass


# ---------- 配置（延迟导入，避免 config 缺字段时整个模块炸掉）----------
def _cfg(name, default=None):
    try:
        import config
        return getattr(config, name, default)
    except Exception:
        return default


class Reranker:
    def __init__(self):
        self.enabled = bool(_cfg("RERANK_ENABLED", False))
        self.backend = str(_cfg("RERANK_BACKEND", "local") or "").strip().lower()
        self.url = _cfg("RERANK_URL", "http://localhost:6006/v1/rerank")
        self.model = _cfg("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
        self.auth = _cfg("RERANK_AUTH", "")
        self.timeout = int(_cfg("RERANK_TIMEOUT", 120) or 120)

        # ---- local 后端专用（旧版全部没读，这里才真正生效）----
        self.local_path = str(_cfg("RERANK_LOCAL_PATH", "") or "").strip()
        self.hf_endpoint = str(_cfg("HF_ENDPOINT", "") or "").strip()
        self.load_timeout = int(_cfg("RERANK_LOAD_TIMEOUT", 180) or 180)
        self.batch_size = int(_cfg("RERANK_BATCH_SIZE", 16) or 16)

        self.failed = False      # 一旦失败就永久降级，避免每条都重试
        self.fail_reason = ""
        self._local = None
        self._lock = threading.Lock()

    # ---------------- HTTP 后端 ----------------
    def _rerank_http(self, query, docs, top_n):
        headers = {"Content-Type": "application/json"}
        if self.auth:
            headers["Authorization"] = f"Bearer {self.auth}"

        payload = {
            "model": self.model,
            "query": query,
            "documents": docs,
            "top_n": top_n,
        }
        r = requests.post(self.url, json=payload, headers=headers,
                          timeout=self.timeout)
        if r.status_code == 404:
            raise RuntimeError(f"404 —— 端点不存在：{self.url}")
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} —— {r.text[:200]}")
        return self._parse(r.json(), docs, top_n)

    def _parse(self, data, docs, top_n):
        """兼容多种返回结构；没有 index 时按返回的文本反查索引"""
        items = None
        for key in ("results", "data", "scores", "ranking"):
            if isinstance(data, dict) and key in data:
                items = data[key]
                break
        if items is None and isinstance(data, list):
            items = data
        if items is None:
            raise RuntimeError(f"无法解析返回结构：{str(data)[:200]}")

        # 服务只回文本不回索引时，用文本反查（旧版把 text 读出来又丢掉了）
        text2idx = {}
        for i, d in enumerate(docs):
            text2idx.setdefault(str(d), i)

        scored = []
        for it in items:
            idx = score = text = None
            if isinstance(it, dict):
                idx = it.get("index", it.get("id"))
                score = it.get("relevance_score",
                               it.get("score", it.get("relevance")))
                text = it.get("document", it.get("text"))
            if idx is None and text is not None:
                idx = text2idx.get(str(text))
            if idx is None:
                continue
            try:
                scored.append((int(idx), float(score or 0)))
            except (TypeError, ValueError):
                continue

        if not scored:
            raise RuntimeError("返回里没有可用的 index/score")
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_n]

    # ---------------- 本地后端 ----------------
    def _model_source(self):
        """决定从哪加载：优先本地目录，绝不偷偷联网"""
        if self.local_path:
            p = Path(self.local_path)
            if p.exists() and any(p.iterdir()):
                return str(p), True
            raise RuntimeError(
                f"RERANK_LOCAL_PATH 指向的目录不存在或为空：{self.local_path}\n"
                f"   👉 先预下载：python predownload_rerank.py")
        # 没填本地路径才退回按模型名找 Hub（可能会下载）
        return self.model, False

    def _load_local(self):
        """加载 cross-encoder。首次加载放子线程 + 超时，避免界面/脚本永久卡死。"""
        if self._local is not None:
            return self._local

        model_src, is_local = self._model_source()

        box = {}

        def worker():
            try:
                # ⚠️ 必须在 import sentence_transformers 之前设镜像，
                #    否则 huggingface_hub 已经按官方源初始化完了。
                if self.hf_endpoint:
                    os.environ.setdefault("HF_ENDPOINT", self.hf_endpoint)
                os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
                # 关掉下载/加载的进度条：控制台里那些
                # 「Loading weights: 100%|████| 393/393」就是它们打的
                os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
                try:
                    from sentence_transformers import CrossEncoder
                    _silence_hf_progress()
                    m = CrossEncoder(model_src)          # 本地目录时完全不联网
                    box["m"] = ("ce", m)
                except ImportError:
                    from FlagEmbedding import FlagReranker
                    _silence_hf_progress()
                    box["m"] = ("flag", FlagReranker(model_src, use_fp16=True))
            except BaseException as e:                   # noqa: BLE001
                box["e"] = e

        t = threading.Thread(target=worker, daemon=True, name="rerank-load")
        t.start()
        t.join(self.load_timeout)

        if t.is_alive():
            raise RuntimeError(
                f"加载 rerank 模型超过 {self.load_timeout}s 未完成"
                f"（{model_src}）。\n"
                f"   多半是在联网下载模型，而下载线程无法被中断。\n"
                f"   👉 先预下载再启动：python predownload_rerank.py\n"
                f"   或调大 config.py 的 RERANK_LOAD_TIMEOUT")
        if "e" in box:
            e = box["e"]
            if isinstance(e, ImportError):
                raise RuntimeError(
                    "本地 rerank 需要装包（会带 torch，约几百 MB）：\n"
                    "  pip install sentence-transformers -i "
                    "https://pypi.tuna.tsinghua.edu.cn/simple")
            raise RuntimeError(f"{type(e).__name__}: {e}")

        self._local = box["m"]
        self._model_src_used = model_src
        if not is_local:
            print(f"   ℹ️ rerank 按模型名加载（{model_src}），"
                  f"若卡住请改用 RERANK_LOCAL_PATH")
        return self._local

    def _rerank_local(self, query, docs, top_n):
        with self._lock:
            kind, m = self._load_local()

        pairs = [[query, d] for d in docs]
        if kind == "ce":
            scores = m.predict(pairs, batch_size=self.batch_size)
        else:
            scores = m.compute_score(pairs)
            if isinstance(scores, (int, float)):
                scores = [scores]

        # CrossEncoder 有时返回 [[s]] 形状，统一压平
        flat = []
        for s in scores:
            if isinstance(s, (list, tuple)):
                s = s[0] if s else 0.0
            flat.append(float(s))

        order = sorted(range(len(flat)), key=lambda i: -flat[i])
        return [(i, flat[i]) for i in order[:top_n]]

    # ---------------- 对外接口 ----------------
    def rank(self, query, docs, top_n):
        """
        返回 [(原始索引, 分数)]，已按分数降序、长度为 top_n。

        返回 None 表示「没重排」（未启用 / 已降级 / 没有文档），
        调用方必须处理 None，不要只 catch 异常。
        """
        if not self.enabled or self.failed or not docs:
            return None

        try:
            if self.backend == "http":
                scored = self._rerank_http(query, docs, top_n)
            elif self.backend == "local":
                scored = self._rerank_local(query, docs, top_n)
            else:
                return None
        except Exception as e:                    # noqa: BLE001
            self.failed = True
            self.fail_reason = f"{type(e).__name__}: {e}"
            print(f"   ⚠️ rerank 失败，降级为纯向量检索：{self.fail_reason[:160]}")
            return None

        scored = [(i, s) for i, s in scored if 0 <= i < len(docs)]
        return scored[:top_n] or None

    def rerank(self, query, docs, top_n):
        """
        兼容旧接口：(重排后的文档列表, 分数列表)。

        任何一个环节降级，分数都返回 None —— 这样 probe() 才能
        区分「真生效」和「静默降级」。
        """
        scored = self.rank(query, docs, top_n)
        if scored is None:
            return docs[:top_n], None
        return ([docs[i] for i, _ in scored], [s for _, s in scored])

    def status(self):
        if not self.enabled:
            return "关闭"
        if self.failed:
            return f"已降级（{self.fail_reason[:40]}）"
        if self.backend == "local":
            src = getattr(self, "_model_src_used", None) or self.local_path \
                or self.model
            return f"local @ {src}"
        return f"{self.backend} @ {self.url}"


def probe():
    """
    独立探测：确认 rerank 是否真的能用。

    ⚠️ 注意：rank()/rerank() 内部失败会【静默降级】并返回 None，
       所以这里必须检查分数是否为 None —— 只靠 try/except 抓不到，
       会误报「可用」（旧版 bug，已修）。
    """
    rk = Reranker()
    print(f"RERANK_ENABLED = {rk.enabled}")
    print(f"RERANK_BACKEND = {rk.backend}")
    print(f"RERANK_MODEL   = {rk.model}")
    if rk.backend == "local":
        print(f"RERANK_LOCAL_PATH = {rk.local_path or '（空 → 会去 Hub 找/下载）'}")
        print(f"HF_ENDPOINT       = {rk.hf_endpoint or '（空 → 官方源）'}")
        print(f"RERANK_LOAD_TIMEOUT = {rk.load_timeout}s")

    if not rk.enabled:
        print("\nℹ️ rerank 当前关闭（改 config.py: RERANK_ENABLED = True 才生效）")
        return

    if rk.backend not in ("local", "http"):
        print(f"\n❌ RERANK_BACKEND = {rk.backend} 不是 local / http，视为关闭")
        return

    if rk.backend == "local":
        print("\n⏳ 加载模型（本地目录则秒开；否则首次会下载 ~1GB）…")
    else:
        print(f"\n⏳ 探测服务：{rk.url}")

    out, scores = rk.rerank("战略规划模型有哪些",
                            ["SWOT分析是一种战略工具",
                             "今天天气不错"], top_n=2)

    if scores is None:
        print(f"\n❌ rerank 未生效：{rk.fail_reason}")
        if rk.backend == "local":
            print("\n   两条路：")
            print("     A. 预下载（推荐，走 HF 镜像，脚本内部设环境变量）：")
            print("        python predownload_rerank.py")
            print("     B. 先关掉：config.py 里 RERANK_ENABLED = False")
        else:
            print("\n   服务没起。要么 docker compose up -d 起 rerank 容器，")
            print("   要么改成本地后端：RERANK_BACKEND = \"local\"")
        return

    print(f"\n✅ rerank 可用（{rk.status()}）")
    for t, s in zip(out, scores):
        print(f"   {s:.4f}  {t[:40]}")


if __name__ == "__main__":
    probe()
