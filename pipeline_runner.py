# -*- coding: utf-8 -*-
"""
后台跑「完整实验」的驱动器（供 Streamlit 的「参数与指标」页调用）

为什么单独一个脚本：
    完整实验 = run_batch → make_reference → run_eval，本机 CPU 下要 40~60 分钟。
    放在 Streamlit 的脚本运行里会阻塞页面（浏览器会超时），所以这里用
    子进程后台跑，把状态写进 data/experiment_status.json、日志写进
    data/experiment.log，前端轮询这两个文件即可。

用法：
    python pipeline_runner.py                 # 跑全套三步
    python pipeline_runner.py --skip-reference # 跳过起草标准答案（reference 已填好时）
    python pipeline_runner.py --note "TOP_K=3"
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
STATUS_FILE = DATA / "experiment_status.json"

STEPS = [
    ("run_batch.py", "生成数据集（检索 + 生成回答）"),
    ("make_reference.py", "起草标准答案"),
    ("run_eval.py", "Ragas 打分"),
]


def write_status(**kw):
    DATA.mkdir(parents=True, exist_ok=True)
    kw["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = STATUS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(kw, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATUS_FILE)          # 原子替换，前端不会读到半截 json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--note", default="", help="写进指标存档的备注")
    ap.add_argument("--skip-reference", action="store_true",
                    help="跳过 make_reference（reference 已人工填好时用）")
    args = ap.parse_args()

    steps = [STEPS[0], STEPS[2]] if args.skip_reference else list(STEPS)

    write_status(state="running", step=0, total=len(steps),
                 message="启动…", started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                 note=args.note)

    for i, (script, label) in enumerate(steps, 1):
        write_status(state="running", step=i, total=len(steps), script=script,
                     message=f"[{i}/{len(steps)}] {label}（{script}）",
                     started_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                     note=args.note)
        print(f"\n{'=' * 60}\n[{i}/{len(steps)}] {label} —— python {script}\n"
              f"{'=' * 60}", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.call([sys.executable, "-u", script], cwd=str(BASE))
        print(f"[{i}/{len(steps)}] {script} 退出码 {rc}，"
              f"用时 {time.perf_counter() - t0:.0f}s", flush=True)
        if rc != 0:
            write_status(state="failed", step=i, total=len(steps), script=script,
                         message=f"{script} 失败（退出码 {rc}），看日志排查",
                         note=args.note)
            return 1

    # ---------- 收尾：把这次的指标写进存档 ----------
    try:
        import config
        import metrics_store

        metrics = metrics_store.metrics_from_scores_csv(config.SCORE_OUT)
        params = {k: getattr(config, k, None) for k in metrics_store.PARAM_KEYS}
        rec = metrics_store.append({
            "mode": "full",
            "version": config.VERSION,
            "source": "ui",
            "n_samples": None,
            "params": params,
            "metrics": metrics,
            "note": args.note or "前端完整实验",
        })
        write_status(state="done", total=len(steps), message="完成，已写入指标存档",
                     metrics=metrics, run_id=rec["run_id"], note=args.note)
        print(f"\n✅ 完成。指标已写入存档：{rec['run_id']} → {metrics}", flush=True)
    except Exception as e:                            # noqa: BLE001
        write_status(state="done", total=len(steps),
                     message=f"跑完了，但写存档失败：{type(e).__name__}: {e}",
                     note=args.note)
        print(f"⚠️ 写存档失败：{e}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
