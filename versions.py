# -*- coding: utf-8 -*-
"""
📦 版本管理：查看 / 清理 data/ 下各版本的产物

用法：
    python versions.py          # 列出所有版本（含参数、回答长度、相似度）
    python versions.py clean    # 删除除最新一版之外的旧产物
    python versions.py cleanall # 清空 data/ 下所有产物（保留 questions.txt）
"""
import sys
import json
import pandas as pd
from pathlib import Path

from config import DATA_DIR, VERSION


def scan():
    """扫描 data/ 下所有版本产物"""
    rows = []
    for xlsx in sorted(DATA_DIR.glob("eval_data_*.xlsx")):
        ver = xlsx.stem.replace("eval_data_", "")
        rec = {"版本": ver, "数据集": xlsx.name}

        mp = xlsx.with_suffix(".json")
        if mp.exists():
            m = json.loads(mp.read_text(encoding="utf-8"))
            rec["生成模型"] = m.get("GEN_MODEL", "-")
            rec["嵌入模型"] = m.get("EMBED_MODEL", "-")
            rec["chunk"] = m.get("CHUNK_SIZE", "-")
            rec["top_k"] = m.get("TOP_K", "-")

        try:
            df = pd.read_excel(xlsx)
            rec["条数"] = len(df)
            lens = [len(str(x).strip()) if str(x).strip() != "nan" else 0
                    for x in df["response"]]
            rec["平均回答字数"] = int(sum(lens) / len(lens)) if lens else 0
            rec["过短条数"] = sum(1 for l in lens if l < 60)
            if "top1_similarity" in df.columns:
                s = df["top1_similarity"].astype(float)
                rec["平均相似度"] = round(float(s.mean()), 3)
        except Exception as e:
            rec["条数"] = f"读取失败 {e}"

        sc = DATA_DIR / f"scores_{ver}.csv"
        if sc.exists():
            try:
                sdf = pd.read_csv(sc)
                for m in ["faithfulness", "answer_relevancy"]:
                    if m in sdf.columns:
                        rec[m] = round(float(sdf[m].mean()), 3)
            except Exception:
                pass
            rec["分数文件"] = sc.name
        else:
            rec["分数文件"] = "（未打分）"

        rows.append(rec)
    return rows


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    rows = scan()

    if cmd == "list":
        if not rows:
            print(f"data/ 下还没有产物，先跑：python run_batch.py")
            return
        df = pd.DataFrame(rows)
        cols = [c for c in ["版本", "条数", "平均回答字数", "过短条数", "平均相似度",
                            "chunk", "top_k", "生成模型", "嵌入模型",
                            "faithfulness", "answer_relevancy"] if c in df.columns]
        print(df[cols].to_string(index=False))
        print(f"\n当前 config.py 的 VERSION = {VERSION}")
        print("\n💡 查看某个版本的完整问答：python show_answers.py v1")
        print("💡 清理旧版本：python versions.py clean")

    elif cmd == "clean":
        if len(rows) <= 1:
            print("只有 0~1 个版本，无需清理")
            return
        keep = rows[-1]["版本"]
        n = 0
        for r in rows[:-1]:
            ver = r["版本"]
            for p in list(DATA_DIR.glob(f"eval_data_{ver}.*")) + \
                     list(DATA_DIR.glob(f"scores_{ver}.*")):
                p.unlink()
                n += 1
                print(f"  🗑️ 删除 {p.name}")
        print(f"\n✅ 清理了 {n} 个文件，保留 {keep}")

    elif cmd == "cleanall":
        n = 0
        for p in list(DATA_DIR.glob("eval_data_*")) + list(DATA_DIR.glob("scores_*")):
            p.unlink()
            n += 1
            print(f"  🗑️ 删除 {p.name}")
        print(f"\n✅ 清空 {n} 个文件")

    else:
        print(f"未知命令：{cmd}（可用：list / clean / cleanall）")


if __name__ == "__main__":
    main()
