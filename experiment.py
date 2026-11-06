# -*- coding: utf-8 -*-
"""
🧪 实验引擎 —— 让「改参数 → 看指标」这件事能在前端完成

两种实验：

① 快速检索实验（run_retrieval_experiment）
   只重新检索 + 重算 context_recall / context_precision，不重新生成回答。
   Ragas 这两个指标只依赖 user_input / retrieved_contexts / reference，
   跟模型写的答案无关，所以可以省掉最慢的生成环节。
   本机实测：10 题约 2~8 分钟（大头是 rerank 的 CPU 计算和云端评委）。

② 完整实验（start_full_experiment）
   后台跑 pipeline_runner.py → run_batch + make_reference + run_eval，
   得到四个指标。40~60 分钟起步，所以放子进程里跑，前端轮询状态文件。

⚠️ 两者都会写 data/metric_runs.jsonl（指标存档），所以历史表里能直接对比。
"""
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OVERRIDE_FILE = DATA / "ui_overrides.json"
STATUS_FILE = DATA / "experiment_status.json"
LOG_FILE = DATA / "experiment.log"


# ---------------------------------------------------------------- 参数
def default_params() -> dict:
    """当前 config 里的参数快照（已包含 ui_overrides.json 的覆盖）"""
    import metrics_store
    import config
    out = {}
    for key in metrics_store.PARAM_KEYS:
        if hasattr(config, key):
            out[key] = getattr(config, key)
    return out


def write_overrides(params: dict) -> Path:
    """把前端参数写进 data/ui_overrides.json，让命令行脚本也用同一套"""
    DATA.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in params.items() if v is not None}
    OVERRIDE_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return OVERRIDE_FILE


def clear_overrides() -> bool:
    if OVERRIDE_FILE.exists():
        OVERRIDE_FILE.unlink()
        return True
    return False


def read_overrides() -> dict:
    if not OVERRIDE_FILE.exists():
        return {}
    try:
        return json.loads(OVERRIDE_FILE.read_text(encoding="utf-8"))
    except Exception:                                  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- 数据集
def load_reference_dataset(path=None):
    """
    读「问题 + 标准答案」，用于检索实验。

    只保留 reference 有内容的行；「文档未提及」的幻觉探针题会被剔除
    （它们不存在"该检索到什么"，硬算 recall 必然 0，属于假警报 ——
     这条口径和 run_eval.py 保持一致）。
    """
    import pandas as pd
    import config

    path = Path(path or config.DATA_OUT)
    if not path.exists():
        raise RuntimeError(
            f"找不到数据集 {path.name}。\n"
            f"检索实验需要「问题 + 标准答案」，请先跑一次：\n"
            f"  python run_batch.py && python make_reference.py")

    df = pd.read_excel(path)
    missing = [c for c in ("user_input", "reference") if c not in df.columns]
    if missing:
        raise RuntimeError(f"数据集缺列：{missing}")

    rows, skipped_no_ref, skipped_probe = [], 0, 0
    for _, r in df.iterrows():
        q = str(r["user_input"]).strip()
        ref = str(r["reference"]).strip()
        if not q:
            continue
        if ref in ("", "nan", "None"):
            skipped_no_ref += 1
            continue
        if "未提及" in ref:
            skipped_probe += 1
            continue
        rows.append({"user_input": q, "reference": ref})
    return rows, {"no_ref": skipped_no_ref, "probe": skipped_probe, "file": path.name}


# ---------------------------------------------------------------- 快速实验
def run_retrieval_experiment(params=None, progress=None, note="",
                             store=True, dataset_path=None):
    """
    只重算检索侧两个指标。返回 record（含 metrics / per_item / n_samples）。
    progress(done, total, text) 可选回调，用于界面上显示进度。
    """
    import metrics_store
    import config
    from retrieval import Retriever

    params = dict(params or default_params())

    def _p(done, total, text):
        if progress:
            progress(done, total, text)

    rows, info = load_reference_dataset(dataset_path)
    if not rows:
        raise RuntimeError(
            "没有可用的「问题 + 标准答案」样本。\n"
            f"（数据集里 reference 为空的 {info['no_ref']} 条、"
            f"幻觉探针 {info['probe']} 条已跳过）\n"
            "请先 python make_reference.py 并人工确认 reference 列。")

    _p(0, len(rows), "加载向量库…")
    retriever = Retriever(
        top_k=params.get("TOP_K", 5),
        pool_k=params.get("RETRIEVE_K", 20),
        use_rerank=bool(params.get("RERANK_ENABLED", False)),
        # 混合检索也一起带上，这样实验页能直接 A/B「BM25 开 vs 关」
        use_hybrid=bool(params.get("HYBRID_ENABLED", False)),
        rrf_k=params.get("HYBRID_RRF_K"),
        vec_weight=params.get("HYBRID_VEC_WEIGHT"),
        bm25_weight=params.get("HYBRID_BM25_WEIGHT"),
        bm25_k1=params.get("BM25_K1"),
        bm25_b=params.get("BM25_B"))

    samples, per_item, t0 = [], [], time.perf_counter()
    for i, row in enumerate(rows, 1):
        _p(i - 1, len(rows), f"检索第 {i}/{len(rows)} 题：{row['user_input'][:20]}…")
        res = retriever.search(row["user_input"])
        samples.append({
            "user_input": row["user_input"],
            "retrieved_contexts": res.docs,
            "reference": row["reference"],
            "response": "",              # 检索指标用不到，留空
        })
        per_item.append({
            "user_input": row["user_input"],
            "score_top1": round(res.scores[0], 4) if res.scores else None,
            "vec_top1": round(res.vec_top1, 4) if res.vec_top1 is not None else None,
            "n_contexts": len(res.docs),
        })
    retrieval_s = time.perf_counter() - t0

    # ---- Ragas 检索侧两指标 ----
    _p(len(rows), len(rows), "调用评委计算 context_recall / precision…")
    import ragas_compat
    ragas_compat.install()

    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import LLMContextRecall, LLMContextPrecisionWithReference

    import run_eval as RE
    client = RE.build_clients()
    llm = RE.build_judge(client)
    run_config = RE.build_run_config()

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(dataset=dataset,
                      metrics=[LLMContextRecall(llm=llm),
                               LLMContextPrecisionWithReference(llm=llm)],
                      run_config=run_config)
    scores = result.to_pandas()
    metrics = metrics_store.metrics_from_scores_df(scores)

    # 逐题分数也带上，方便在界面上定位是哪道题拉低了指标
    for key in ("context_recall", "context_precision"):
        col = metrics_store.pick_column(scores, key)
        if not col:
            continue
        for i, item in enumerate(per_item):
            if i < len(scores):
                item[key] = _num_or_none(scores[col].iloc[i])

    elapsed = time.perf_counter() - t0
    record = {
        "mode": "retrieval",
        "version": config.VERSION,
        "source": "ui",
        "n_samples": len(samples),
        "judge": params.get("JUDGE_MODEL"),
        "params": params,
        "metrics": metrics,
        "note": note or "前端快速检索实验",
        "detail": {
            "retrieval_seconds": round(retrieval_s, 1),
            "total_seconds": round(elapsed, 1),
            "dataset": info,
            "per_item": per_item,
        },
    }
    if store:
        record = metrics_store.append(record)
    _p(len(rows), len(rows), f"完成，用时 {elapsed:.0f}s")
    return record


def _num_or_none(v):
    try:
        import math
        f = float(v)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- 完整实验
def start_full_experiment(params=None, note="", skip_reference=False) -> dict:
    """后台启动完整实验；返回当前状态。防重复启动。"""
    status = read_status()
    if status.get("state") == "running" and not _stale(status):
        raise RuntimeError("已经有一个完整实验在跑，等它结束或先看日志")

    if params:
        write_overrides(params)

    DATA.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(BASE / "pipeline_runner.py"),
           "--note", note or "前端完整实验"]
    if skip_reference:
        cmd.append("--skip-reference")

    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) \
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    log = LOG_FILE.open("a", encoding="utf-8")
    log.write(f"\n\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} 启动："
              f"{' '.join(cmd[1:])} =====\n")
    log.flush()
    proc = subprocess.Popen(cmd, cwd=str(BASE), stdout=log,
                            stderr=subprocess.STDOUT, **kwargs)

    # 先把状态写成 running，避免前端在子进程写状态之前读到旧的 done
    STATUS_FILE.write_text(json.dumps({
        "state": "running", "step": 0, "total": 2 if skip_reference else 3,
        "message": "已启动，等待子进程汇报…", "pid": proc.pid,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return read_status()


def read_status() -> dict:
    if not STATUS_FILE.exists():
        return {"state": "idle", "message": "还没有跑过完整实验"}
    try:
        return json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except Exception:                                   # noqa: BLE001
        return {"state": "unknown", "message": "状态文件损坏"}


def is_running() -> bool:
    st = read_status()
    return st.get("state") == "running" and not _stale(st)


def _stale(status, minutes=180) -> bool:
    """状态停在 running 但很久没更新 → 认为进程已经死了"""
    ts = status.get("updated_at")
    if not ts:
        return True
    try:
        t = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        return True
    return (time.time() - t) > minutes * 60


def read_log_tail(lines=60) -> str:
    if not LOG_FILE.exists():
        return ""
    try:
        with LOG_FILE.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except Exception as e:                              # noqa: BLE001
        return f"（日志读取失败：{e}）"
