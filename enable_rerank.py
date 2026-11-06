# -*- coding: utf-8 -*-
"""
一键开启 rerank（local 后端，不需要 Docker）

帮你做完三件事，避免手改 config 漏项：
  ① 下载模型（复用 predownload_rerank.py，走 HF 镜像 + 关 Xet）
  ② 自动改 config.py（只精确替换那几行，其余内容原样保留 + 自动备份）
  ③ 验证是否真的生效（检查分数不为 None，防止「显示开了其实在降级」）

用法：
    python enable_rerank.py                            # 默认 bge-reranker-v2-m3
    python enable_rerank.py BAAI/bge-reranker-base     # 更小的（约 400MB）
    python enable_rerank.py --skip-download            # 已下载过，只改配置
"""
import re
import shutil
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
CONFIG = BASE / "config.py"

DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

args = sys.argv[1:]
SKIP_DOWNLOAD = "--skip-download" in args
if SKIP_DOWNLOAD:
    # 注意：这里必须移除【字符串】，不能写成 args.remove(SKIP_DOWNLOAD)
    # —— SKIP_DOWNLOAD 是布尔值，remove(True) 会抛 ValueError
    args.remove("--skip-download")

model = args[0] if args else DEFAULT_MODEL
if "/" not in model and model.startswith("bge-"):
    model = f"BAAI/{model}"

short = model.split("/")[-1]
local_dir = BASE / "models" / short

print("=" * 64)
print("一键开启 rerank（local 后端）")
print("=" * 64)
print(f"  模型     : {model}")
print(f"  本地目录 : {local_dir}")
print(f"  需要下载 : {'否（--skip-download）' if SKIP_DOWNLOAD else '是'}")
print("=" * 64)
print()

# 先检查运行 rerank 真正需要的包（注意：和下载用的 huggingface_hub 不是同一个）
#   · huggingface_hub       —— 只负责下载
#   · sentence-transformers —— 负责加载并跑模型，会带 torch（几百 MB）
try:
    import sentence_transformers  # noqa: F401
    print("✅ sentence-transformers 已安装")
except ImportError:
    try:
        import FlagEmbedding  # noqa: F401
        print("✅ FlagEmbedding 已安装")
    except ImportError:
        print("❌ 缺少运行 rerank 的包，先装（会带 torch，几百 MB）：")
        print("   pip install sentence-transformers "
              "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        print("\n   装完再重新跑本脚本。")
        sys.exit(1)
print()

# ---------------------------------------------------------------- ① 下载
# 直接复用 predownload_rerank.py，避免两份下载逻辑各自踩坑
# （镜像要在 import 前设、Xet 要关、目录不完整要能续下）
import predownload_rerank as P  # noqa: E402

P.enable_mirror(P.DEFAULT_MIRROR)

if SKIP_DOWNLOAD:
    if not P.dir_is_complete(local_dir):
        print(f"❌ 目录不存在或不完整：{local_dir}")
        print("   去掉 --skip-download 重新跑")
        sys.exit(1)
    print(f"⏭ 跳过下载，使用已有目录：{local_dir}")
else:
    if P.dir_is_complete(local_dir):
        print(f"📂 已存在完整模型，跳过下载：{local_dir}")
    else:
        print(f"⏳ 下载中（镜像 {P.DEFAULT_MIRROR}，超时 600s）…")
        print("   首次下载视网络可能几分钟，请耐心\n")
        t0 = time.perf_counter()
        try:
            from huggingface_hub import snapshot_download
            try:
                snapshot_download(repo_id=model, local_dir=str(local_dir))
            except TypeError:
                snapshot_download(repo_id=model, local_dir=str(local_dir),
                                  local_dir_use_symlinks=False)
        except Exception as e:                    # noqa: BLE001
            print(f"\n❌ 下载失败：{type(e).__name__}: {e}")
            print("\n三个退路：")
            print("  1. 换更小的模型：python enable_rerank.py "
                  "BAAI/bge-reranker-base")
            print("  2. 单独跑带诊断的下载：python predownload_rerank.py")
            print("  3. 放弃：config.py 里 RERANK_ENABLED = False")
            sys.exit(1)
        print(f"\n✅ 下载完成，用时 {time.perf_counter() - t0:.0f}s")

# ---------------------------------------------------------------- ② 改 config
print()
print("─" * 64)
print("修改 config.py")
print("─" * 64)

if not CONFIG.exists():
    print(f"❌ 找不到 {CONFIG}")
    sys.exit(1)

src = CONFIG.read_text(encoding="utf-8")
backup = BASE / f"config.py.bak.{time.strftime('%m%d_%H%M%S')}"
shutil.copy2(CONFIG, backup)


def set_line(text, key, value):
    """精确替换某一行配置；不存在就追加到文件末尾"""
    pat = re.compile(rf"^(\s*){re.escape(key)}\s*=\s*(.+?)\s*(#.*)?$",
                     re.MULTILINE)
    new = f"{key} = {value}"
    if pat.search(text):
        return pat.sub(lambda m: new, text, count=1), True
    return text.rstrip() + f"\n\n{new}\n", False


for key, val in [("RERANK_ENABLED", "True"),
                 ("RERANK_BACKEND", '"local"'),
                 ("RERANK_MODEL", f'"{model}"'),
                 ("RERANK_LOCAL_PATH", f'r"{local_dir}"')]:
    src, hit = set_line(src, key, val)
    print(f"  {'改写' if hit else '追加'} {key} = {val}")

# RETRIEVE_K 必须 >= TOP_K，顺手兜底
m = re.search(r"^TOP_K\s*=\s*(\d+)", src, re.MULTILINE)
top_k = int(m.group(1)) if m else 5
m2 = re.search(r"^RETRIEVE_K\s*=\s*(\d+)", src, re.MULTILINE)
retrieve_k = int(m2.group(1)) if m2 else 20
if retrieve_k < top_k:
    src, _ = set_line(src, "RETRIEVE_K", str(max(top_k * 4, 20)))
    print(f"  ⚠️ RETRIEVE_K({retrieve_k}) < TOP_K({top_k})，已自动调大")

CONFIG.write_text(src, encoding="utf-8")
print(f"\n💾 已保存（原文件备份 → {backup.name}）")

# ---------------------------------------------------------------- ③ 验证
print()
print("─" * 64)
print("验证是否真的生效")
print("─" * 64)

# 重新加载 config（清掉旧缓存）
for mod in ("config", "rerank"):
    sys.modules.pop(mod, None)
sys.path.insert(0, str(BASE))

try:
    import rerank as _rk
    rk = _rk.Reranker()
    print(f"  RERANK_ENABLED = {rk.enabled}")
    print(f"  RERANK_BACKEND = {rk.backend}")
    print(f"  本地路径完整   = {P.dir_is_complete(local_dir)}")

    if not rk.enabled:
        print("\n❌ 仍是关闭状态，检查 config.py")
        sys.exit(1)

    print("\n  ⏳ 加载模型并试跑一次…")
    out, scores = rk.rerank("战略规划模型有哪些",
                            ["SWOT分析是一种战略工具",
                             "今天天气不错"], top_n=2)

    if scores is None:
        print(f"\n❌ 未生效：{rk.fail_reason}")
        print("\n   会走「自动降级」，界面和评测照常能跑，只是没用到 rerank。")
        sys.exit(1)

    print("\n✅ rerank 生效！排序结果：")
    for t, s in zip(out, scores):
        print(f"     {s:.4f}  {t[:40]}")
    print("\n   相关文档排在最前 → 说明工作正常")

except SystemExit:
    raise
except Exception as e:                            # noqa: BLE001
    print(f"\n❌ 验证出错：{type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---------------------------------------------------------------- 下一步
print()
print("=" * 64)
print("✅ 完成。接下来的操作")
print("=" * 64)
print("""
【A. 界面上看效果】
   streamlit run app.py
   左上角 rerank 应显示「开」（显示「降级⚠️」说明没生效）
   引用原文的顺序会和之前不同

【B. 跑评测对比 —— 这才是加 rerank 的目的】
   1. python run_batch.py        ← 必须重跑（contexts 变了）
   2. python make_reference.py   ← 门禁仍用余弦相似度，不受 rerank 影响
   3. python run_eval.py
   4. python compare.py

【重点看哪个数】
   只盯 context_precision
   · 涨了 → rerank 有效
   · 没变 → 瓶颈在召回而非排序，该换 embedding 而不是 rerank
   · recall 基本不会变（rerank 只改顺序不改召回集合）
""")
