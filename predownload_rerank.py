# -*- coding: utf-8 -*-
"""
⬇️ 预下载 rerank 模型（本地后端专用）

为什么必须单独一步：
    如果等到检索时再让 sentence-transformers 去下载，会出两个问题：
      ① 国内直连 huggingface.co 基本必超时（WinError 10060）
      ② 下载线程【无法被打断】，界面会永远卡在「检索中...」
    所以把下载挪到离线一步完成，之后只读本地目录，完全不联网。

用法：
    python predownload_rerank.py                        # 默认 bge-reranker-v2-m3
    python predownload_rerank.py BAAI/bge-reranker-base # 更小（约 570MB）
    python predownload_rerank.py --mirror ""            # 走官方源（国内一般会失败）
    python predownload_rerank.py --dest .\models\rr     # 指定落盘目录
    python predownload_rerank.py --direct               # 强制用「直连分块下载」

两种下载方式（先试 ①，失败自动退到 ②）：
    ① huggingface_hub.snapshot_download  —— 正常网络下最省事
    ② 直连 HTTP + Range 断点续传          —— 绕过 hub 的 LFS / Xet 客户端

    ⚠️ 为什么需要 ②：国内镜像常出现「小文件能下、大权重文件卡在 0 字节」——
       hub 的 LFS 下载会走 CDN/CAS 端点（可能 401，也可能被限速）。
       实测同一台机器：hub 卡死在 0 字节；改用普通 HTTP Range 直连能跑到
       2.3 MB/s（2.27GB 约 16 分钟），而且断了能续。

    ③ 两个都下不动时，用 ModelScope（实测 33 MB/s）：
       pip install modelscope
       python -c "from modelscope import snapshot_download; \
snapshot_download('BAAI/bge-reranker-v2-m3', local_dir=r'models\bge-reranker-v2-m3')"
       然后照常 python rerank.py 验证。

⚠️ 关键顺序（旧版本踩过的坑）：
    HF_ENDPOINT 必须在【import huggingface_hub / sentence_transformers 之前】
    设进 os.environ。因为这两个库在 import 时就把端点读成常量了，
    之后再 os.environ[...] = ... 完全无效，会照旧去连 huggingface.co 然后超时。
    所以本脚本先解析参数 → 设环境变量 → 才 import 这两个库。
"""
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MIRROR = "https://hf-mirror.com"

# 判断目录里是不是已经有完整模型（只看"非空"会把下了一半的当成完成）
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".onnx", ".gguf")


def parse_args(argv):
    args = [a for a in argv]
    mirror = DEFAULT_MIRROR
    dest = None
    model = None
    force_direct = False

    i = 0
    while i < len(args):
        a = args[i]
        if a == "--mirror":
            if i + 1 >= len(args):
                raise SystemExit("--mirror 后面要跟镜像地址（想用官方源就写 \"\"）")
            mirror = args[i + 1]
            i += 2
            continue
        if a == "--dest":
            if i + 1 >= len(args):
                raise SystemExit("--dest 后面要跟目标目录")
            dest = Path(args[i + 1]).expanduser().resolve()
            i += 2
            continue
        if a == "--direct":
            force_direct = True
            i += 1
            continue
        if a.startswith("-"):
            raise SystemExit(f"未知参数：{a}")
        model = model or a
        i += 1

    return model or DEFAULT_MODEL, mirror, dest, force_direct


def enable_mirror(mirror):
    """必须在任何 HF 相关 import 之前调用"""
    if mirror:
        os.environ["HF_ENDPOINT"] = mirror
    # 下载大文件时默认超时偏短，放宽（单位秒）
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    # ⚠️ 关键：huggingface_hub 0.3x+ 默认走 Xet 协议下载大文件，
    #    而 hf-mirror 这类镜像的 CAS 端点没有对应鉴权，会报
    #    401 Unauthorized (cas-server.xethub.hf.co)。
    #    关掉 Xet，退回传统 LFS/HTTP 下载，镜像才可用。
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def dir_is_complete(target: Path) -> bool:
    if not target.exists():
        return False
    files = [p for p in target.rglob("*") if p.is_file()]
    if not files:
        return False
    has_cfg = any(p.name in ("config.json", "config_sentence_transformers.json")
                  for p in files)
    has_weight = any(p.suffix in WEIGHT_SUFFIXES and p.stat().st_size > 1_000_000
                     for p in files)
    return has_cfg and has_weight


# ---------------------------------------------------------------- ① hub 下载
def download_via_hub(model, target: Path):
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(repo_id=model, local_dir=str(target))
    except TypeError:
        # 新版 hub 已移除 local_dir_use_symlinks
        snapshot_download(repo_id=model, local_dir=str(target),
                          local_dir_use_symlinks=False)


# ---------------------------------------------------------------- ② 直连下载
def _list_files(endpoint, model, revision="main"):
    import requests
    url = f"{endpoint.rstrip('/')}/api/models/{model}/tree/{revision}?recursive=true"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return [f for f in r.json()
            if f.get("type") == "file"
            and not str(f.get("path", "")).startswith("assets/")]


def download_via_direct(model, target: Path, endpoint=None, revision="main",
                        retries=40):
    """
    普通 HTTP + Range 断点续传，绕过 hub 的 LFS/Xet 客户端。
    每个文件写完校验大小；中途断开就带 Range 续传，最多重试 retries 次。
    """
    import requests

    endpoint = (endpoint or DEFAULT_MIRROR).rstrip("/")
    files = _list_files(endpoint, model, revision)
    total_bytes = sum(int(f.get("size") or 0) for f in files)
    print(f"  共 {len(files)} 个文件 / {total_bytes / 1e9:.2f} GB")
    print(f"  端点：{endpoint}（直连 HTTP，可断点续传）\n")

    target.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    for f in files:
        rel = f["path"]
        size = int(f.get("size") or 0)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        if size and dest.exists() and dest.stat().st_size == size:
            print(f"  ✓ 已完整，跳过：{rel}")
            continue

        url = f"{endpoint}/{model}/resolve/{revision}/{rel}"
        got = dest.stat().st_size if dest.exists() else 0
        if got and size and got > size:
            got = 0                      # 文件比预期大，重下
        attempt = 0
        last_report = 0.0

        while True:
            headers = {"Range": f"bytes={got}-"} if got else {}
            try:
                with requests.get(url, headers=headers, stream=True,
                                  timeout=(15, 60)) as r:
                    if r.status_code == 416:      # 已经下完
                        break
                    if r.status_code not in (200, 206):
                        raise RuntimeError(f"HTTP {r.status_code}")
                    mode = "ab" if (got and r.status_code == 206) else "wb"
                    if mode == "wb":
                        got = 0
                    with open(dest, mode) as fh:
                        for chunk in r.iter_content(262144):
                            if not chunk:
                                continue
                            fh.write(chunk)
                            got += len(chunk)
                            if size and time.perf_counter() - last_report > 5:
                                pct = got / size * 100
                                spd = got / max(time.perf_counter() - t_start, .1)
                                print(f"    {rel}  {pct:5.1f}%  "
                                      f"({got / 1e6:.0f}/{size / 1e6:.0f} MB, "
                                      f"{spd / 1e6:.1f} MB/s)", end="\r")
                                last_report = time.perf_counter()
                    if size and got < size:
                        raise RuntimeError(f"连接提前结束（{got}/{size} 字节）")
                print(f"  ✓ {rel}  {got / 1e6:.1f} MB" + " " * 30)
                break
            except Exception as e:                    # noqa: BLE001
                attempt += 1
                if attempt > retries:
                    raise RuntimeError(
                        f"{rel} 下载失败（已重试 {retries} 次）："
                        f"{type(e).__name__}: {e}")
                wait = min(2 * attempt, 15)
                print(f"\n    ⚠️ {rel} 中断（{type(e).__name__}），"
                      f"{wait}s 后续传（第 {attempt}/{retries} 次）…")
                time.sleep(wait)
                got = dest.stat().st_size if dest.exists() else 0

    print(f"\n  直连下载完成，用时 {time.perf_counter() - t_start:.0f}s")


def main():
    model, mirror, dest, force_direct = parse_args(sys.argv[1:])
    short = model.split("/")[-1]
    target = dest or (BASE / "models" / short)

    # ---- 第一件事：设镜像，早于所有 HF import ----
    enable_mirror(mirror)

    print("=" * 64)
    print("预下载 rerank 模型")
    print("=" * 64)
    print(f"  模型     : {model}")
    print(f"  镜像     : {mirror or '（官方源 huggingface.co）'}")
    print(f"  落盘目录 : {target}")
    print(f"  下载方式 : {'直连（--direct）' if force_direct else '先 hub，失败自动转直连'}")
    print("=" * 64)

    # 运行 rerank 需要 sentence-transformers（或 FlagEmbedding）
    have_st = have_flag = False
    try:
        import sentence_transformers  # noqa: F401
        have_st = True
    except ImportError:
        try:
            import FlagEmbedding  # noqa: F401
            have_flag = True
        except ImportError:
            pass
    if not (have_st or have_flag):
        print("\n❌ 缺运行包，先装（会带 torch，几百 MB）：")
        print("   pip install sentence-transformers "
              "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        return 1
    print(f"\n✅ 运行包就绪：{'sentence-transformers' if have_st else 'FlagEmbedding'}")

    if dir_is_complete(target):
        print(f"\n📂 已存在完整模型，跳过下载：{target}")
    else:
        if target.exists() and any(target.iterdir()):
            print(f"\n🔁 目录已存在但模型不完整（上次可能下到一半），继续补下…")

        ok = False
        if force_direct:
            try:
                download_via_direct(model, target, mirror or DEFAULT_MIRROR)
                ok = True
            except Exception as e:                # noqa: BLE001
                print(f"\n❌ 直连下载失败：{type(e).__name__}: {e}")
        else:
            print(f"\n① 先试 huggingface_hub 下载（超时 600s）…")
            t0 = time.perf_counter()
            try:
                download_via_hub(model, target)
                ok = True
                print(f"✅ hub 下载完成，用时 {time.perf_counter() - t0:.0f}s")
            except Exception as e:                # noqa: BLE001
                print(f"\n⚠️ hub 下载失败（{type(e).__name__}: "
                      f"{str(e)[:120]}）")
                if dir_is_complete(target):
                    ok = True
                else:
                    print("② 改用直连 HTTP（断点续传，绕过 LFS/Xet 客户端）…")
                    try:
                        download_via_direct(model, target,
                                            mirror or DEFAULT_MIRROR)
                        ok = True
                    except Exception as e2:       # noqa: BLE001
                        print(f"\n❌ 直连下载也失败：{type(e2).__name__}: {e2}")

        if not ok and not dir_is_complete(target):
            print("\n三个退路：")
            print("  1. 换更小的模型：python predownload_rerank.py "
                  "BAAI/bge-reranker-base")
            print("  2. 换镜像：python predownload_rerank.py --mirror "
                  "https://hf-mirror.com")
            print("  3. 放弃 rerank：config.py 里 RERANK_ENABLED = False")
            print("\n   （rerank 下载失败不影响主流程，运行时会自动降级）")
            return 1

    # 真的能加载才算成功（只下载不验证，等于没验证）
    print("\n⏳ 验证能否离线加载（不联网）…")
    try:
        from sentence_transformers import CrossEncoder
        t0 = time.perf_counter()
        m = CrossEncoder(str(target))
        scores = m.predict([["战略规划模型有哪些", "SWOT分析是一种战略工具"],
                            ["战略规划模型有哪些", "今天天气不错"]])
        flat = [float(s[0]) if isinstance(s, (list, tuple)) else float(s)
                for s in scores]
        print(f"✅ 加载成功（{time.perf_counter() - t0:.1f}s）")
        print(f"   相关句得分 {flat[0]:.4f} / 无关句得分 {flat[1]:.4f}")
        if flat[0] <= flat[1]:
            print("   ⚠️ 相关句得分没有更高，模型或模型名可能不对")
    except Exception as e:                        # noqa: BLE001
        print(f"❌ 加载失败：{type(e).__name__}: {e}")
        print("   下载的目录可能不完整，删掉后重新跑本脚本")
        return 1

    print("\n" + "=" * 64)
    print("✅ 完成。把下面四行填进 config.py（照抄即可）")
    print("=" * 64)
    print('RERANK_ENABLED    = True')
    print('RERANK_BACKEND    = "local"')
    print(f'RERANK_MODEL      = "{model}"')
    print(f'RERANK_LOCAL_PATH = r"{target}"')
    print("=" * 64)
    print("\n下一步：")
    print("  python rerank.py            # 验证真的生效（应打印 ✅ 和两行分数）")
    print("  python check_pipeline.py    # 全链路体检")
    print("\n⚠️ 想对比 rerank 的效果，必须重新跑 run_batch.py：")
    print("   rerank 改变了 retrieved_contexts，旧数据集里的上下文不能复用。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
