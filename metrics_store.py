# -*- coding: utf-8 -*-
"""
🗃️ 指标存档 —— 把每次评测的结果（参数 + 指标）落到 data/metric_runs.jsonl

为什么用 jsonl 而不是直接堆 xlsx：
    · 追加式写入，不会因为并发/中断毁掉整张表
    · 一条记录自带参数快照，天然支持「同一批题、不同参数」的对比
    · 人和脚本都能读，也可以随时导出成 csv

记录结构（一行一条）：
    {
      "run_id": "20261102-181500-3f2a",
      "ts": "2026-11-02 18:15:00",
      "mode": "full" | "retrieval",
      "version": "v1",
      "source": "ui" | "cli",
      "n_samples": 8,
      "judge": "gemma4:31b-cloud",
      "params": {"TOP_K": 5, "RETRIEVE_K": 20, "RERANK_ENABLED": true, ...},
      "metrics": {"faithfulness": 0.974, "context_recall": 1.0, ...},
      "note": "改了 TOP_K=3"
    }

⚠️ 只有 retrieval 模式的记录才有 context_recall / context_precision 两个指标
   （它不重新生成回答，所以 faithfulness / answer_relevancy 没有意义，留空）。
"""
import json
import time
import uuid
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
RUNS_FILE = DATA_DIR / "metric_runs.jsonl"
EXPORT_FILE = DATA_DIR / "metric_runs.csv"

# 四指标的展示顺序 / 中文名
METRIC_NAMES = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_recall": "context_recall",
    "context_precision": "context_precision",
}
METRIC_LABELS = {
    "faithfulness": "忠实度（有没有编）",
    "answer_relevancy": "相关性（有没有答到点）",
    "context_recall": "召回（该检的漏没漏）",
    "context_precision": "精确（检到的排得靠前吗）",
}

# ragas 列名兼容（不同版本叫法不一样）
COLUMN_ALIASES = {
    "faithfulness": ["faithfulness"],
    "answer_relevancy": ["answer_relevancy", "response_relevancy"],
    "context_recall": ["context_recall", "llm_context_recall"],
    "context_precision": ["llm_context_precision_with_reference",
                          "context_precision", "llm_context_precision"],
}

# 参数快照里要记的键（顺序即展示顺序）
PARAM_KEYS = ["GEN_MODEL", "EMBED_MODEL", "JUDGE_MODEL", "TOP_K", "RETRIEVE_K",
              "RERANK_ENABLED", "RERANK_BACKEND", "RERANK_MODEL",
              "HYBRID_ENABLED", "HYBRID_VEC_WEIGHT", "HYBRID_BM25_WEIGHT",
              "HYBRID_RRF_K", "BM25_K1", "BM25_B",
              "CHUNK_SIZE", "CHUNK_OVERLAP", "TEMPERATURE", "NUM_CTX",
              "SIM_THRESHOLD", "COLLECTION"]


def _ensure_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def new_run_id():
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


def append(record: dict) -> dict:
    """追加一条记录；自动补 run_id / ts，返回补全后的记录"""
    _ensure_dir()
    record = dict(record)
    record.setdefault("run_id", new_run_id())
    record.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
    record.setdefault("source", "ui")
    with RUNS_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_all() -> list[dict]:
    if not RUNS_FILE.exists():
        return []
    out = []
    for line in RUNS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue           # 半行/坏行直接跳过，不让一条坏数据毁掉整表
    return out


def latest() -> dict | None:
    runs = load_all()
    return runs[-1] if runs else None


def to_dataframe(runs: list[dict] | None = None) -> pd.DataFrame:
    """把记录摊平成一张表：每个指标一列，常用参数各一列"""
    runs = load_all() if runs is None else runs
    rows = []
    for r in runs:
        params = r.get("params") or {}
        row = {
            "run_id": r.get("run_id"),
            "时间": r.get("ts"),
            "模式": r.get("mode"),
            "版本": r.get("version"),
            "来源": r.get("source"),
            "样本数": r.get("n_samples"),
            "备注": r.get("note", ""),
        }
        for key in PARAM_KEYS:
            if key in params:
                row[key] = params[key]
        for metric in METRIC_NAMES:
            row[metric] = (r.get("metrics") or {}).get(metric)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # 指标列保持数值型（全空时会变成 object，这里强制一下）
    for metric in METRIC_NAMES:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def pick_column(df: pd.DataFrame, key: str) -> str | None:
    for c in COLUMN_ALIASES.get(key, [key]):
        if c in df.columns:
            return c
    return None


def metrics_from_scores_df(df: pd.DataFrame) -> dict:
    """从 ragas 结果表（to_pandas()）里算四指标均值"""
    out = {}
    for key in METRIC_NAMES:
        col = pick_column(df, key)
        if col and pd.api.types.is_numeric_dtype(df[col]):
            out[key] = round(float(df[col].mean()), 4)
    return out


def metrics_from_scores_csv(path) -> dict:
    """从 ragas 导出的 scores_*.csv 里算四指标均值"""
    path = Path(path)
    if not path.exists():
        return {}
    return metrics_from_scores_df(pd.read_csv(path))


def import_scores_csv(path, version=None, params=None, note="从已有分数文件导入",
                      mode="full") -> dict | None:
    """把上一次 run_eval.py 的结果补录进存档（这样 v1 也能出现在历史里）"""
    path = Path(path)
    metrics = metrics_from_scores_csv(path)
    if not metrics:
        return None
    version = version or path.stem.replace("scores_", "")
    record = {
        "mode": mode,
        "version": version,
        "source": "cli",
        "n_samples": _count_rows(path),
        "params": params or {},
        "metrics": metrics,
        "note": note,
    }
    return append(record)


def _count_rows(path) -> int | None:
    try:
        return int(len(pd.read_csv(path)))
    except Exception:
        return None


def export_csv() -> Path:
    """把存档导出成 csv（Excel 友好，utf-8-sig）"""
    _ensure_dir()
    df = to_dataframe()
    df.to_csv(EXPORT_FILE, index=False, encoding="utf-8-sig")
    return EXPORT_FILE


def clear() -> None:
    if RUNS_FILE.exists():
        RUNS_FILE.unlink()
