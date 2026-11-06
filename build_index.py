# -*- coding: utf-8 -*-
"""
① 建向量库：读 docs/ 下所有文档 → 切块 → embedding → 存 vector_store/

【无需 chromadb】纯 numpy 存储 + 暴力余弦检索，小知识库（几千块）毫秒级。

只在改了 CHUNK_SIZE / CHUNK_OVERLAP / EMBED_MODEL 之后需要重跑。
"""
import re
import json
import html as html_lib
import requests
import numpy as np
from pathlib import Path
from html.parser import HTMLParser

from config import (BASE, DOCS_DIR, DB_DIR, OLLAMA_URL, EMBED_MODEL,
                    CHUNK_SIZE, CHUNK_OVERLAP, COLLECTION, VERSION)

# 命令行开关：
#   python build_index.py --no-bump     本次不自动递增版本号
#   python build_index.py --version v9  手动指定版本号（会直接写进 config.py）
import sys as _sys
AUTO_BUMP_VERSION = getattr(
    __import__("config"), "AUTO_BUMP_VERSION", True)
if "--no-bump" in _sys.argv:
    AUTO_BUMP_VERSION = False
FORCE_VERSION = None
if "--version" in _sys.argv:
    _i = _sys.argv.index("--version")
    if _i + 1 < len(_sys.argv):
        FORCE_VERSION = _sys.argv[_i + 1]

# 支持的文件后缀
SUPPORTED = {".txt", ".md", ".pdf", ".docx", ".pptx", ".html", ".htm"}


# ---------- HTML 提取器（纯标准库，无需安装任何包）----------
class _HTMLText(HTMLParser):
    """把 HTML 里的正文抽出来，跳过 script/style/noscript"""
    SKIP = {"script", "style", "noscript", "svg", "head"}
    BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4",
             "h5", "h6", "section", "article", "td", "th"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip_depth += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth == 0 and data.strip():
            self.parts.append(data.strip() + " ")

    def text(self):
        txt = "".join(self.parts)
        txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
        txt = re.sub(r"\n\s*\n\s*\n+", "\n\n", txt)
        return txt.strip()


def read_html(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "gb18030", "utf-16"):
        try:
            src = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        src = raw.decode("utf-8", errors="ignore")

    # <title> 单独保留，往往是文档标题
    m = re.search(r"<title[^>]*>(.*?)</title>", src, re.S | re.I)
    title = html_lib.unescape(m.group(1)).strip() if m else ""

    p = _HTMLText()
    p.feed(src)
    body = p.text()
    return f"{title}\n\n{body}" if title else body


def read_pptx(path: Path) -> str:
    """需要 python-pptx；提取每页文字 + 表格 + 备注"""
    from pptx import Presentation
    prs = Presentation(str(path))
    out = []
    for i, slide in enumerate(prs.slides, 1):
        buf = [f"[第{i}页]"]
        for shape in slide.shapes:
            if shape.has_text_frame:
                t = "\n".join(par.text for par in shape.text_frame.paragraphs
                              if par.text.strip())
                if t.strip():
                    buf.append(t)
            # 表格
            if getattr(shape, "has_table", False) and shape.has_table:
                for row in shape.table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        buf.append(" | ".join(cells))
        # 备注页
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            note = slide.notes_slide.notes_text_frame.text.strip()
            if note:
                buf.append(f"（备注：{note}）")
        if len(buf) > 1:
            out.append("\n".join(buf))
    return "\n\n".join(out)


def read_docx(path: Path) -> str:
    """需要 python-docx；段落 + 表格"""
    from docx import Document
    d = Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for tb in d.tables:
        for row in tb.rows:
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                parts.append(" | ".join(cells))
    return "\n".join(parts)


# ---------- 1. 读取文档 ----------
def _rel_name_dir(p: Path):
    """
    返回 (相对 docs 的路径, 文件名, 所在文件夹)。

    为什么要记文件夹：前端支持「只搜某个文件夹」，
    而 chunk 的元数据里必须带上这个信息，否则检索时无法按目录过滤。
    根目录下的文件 dir_of 为 ""。
    """
    try:
        rel = p.relative_to(DOCS_DIR).as_posix()
    except ValueError:
        rel = p.name
    dir_of = rel.rsplit("/", 1)[0] if "/" in rel else ""
    return rel, p.name, dir_of


def load_docs():
    texts = []
    if not DOCS_DIR.exists():
        DOCS_DIR.mkdir(parents=True)
        print(f"📁 已创建 {DOCS_DIR}，把你的文档放进去后重新运行")
        return texts

    # 先汇报一下 docs 里都有什么，不支持的后缀会明确列出来
    all_files = [p for p in DOCS_DIR.rglob("*") if p.is_file()]
    global ALL_FILES
    ALL_FILES = all_files          # 供 main() 最后做缺失汇总
    unsupported = [p for p in all_files if p.suffix.lower() not in SUPPORTED]
    if unsupported:
        print(f"  ⚠️ 有 {len(unsupported)} 个不支持的文件被跳过：")
        for p in unsupported[:10]:
            print(f"     - {p.name}  （{p.suffix}）")

    # ① txt / md —— 无需任何依赖
    for p in list(DOCS_DIR.rglob("*.txt")) + list(DOCS_DIR.rglob("*.md")):
        rel, name, dir_of = _rel_name_dir(p)
        try:
            texts.append((rel, name, dir_of, p.read_text(encoding="utf-8")))
        except UnicodeDecodeError:
            texts.append((rel, name, dir_of,
                          p.read_text(encoding="gb18030", errors="ignore")))

    # ② html / htm —— 纯标准库，无需安装
    htmls = list(DOCS_DIR.rglob("*.html")) + list(DOCS_DIR.rglob("*.htm"))
    for p in htmls:
        try:
            texts.append((*_rel_name_dir(p), read_html(p)))
        except Exception as e:
            print(f"  ⚠️ HTML 读取失败 {p.name}: {e}")

    # ③ pdf —— 需 pypdf
    pdfs = list(DOCS_DIR.rglob("*.pdf"))
    if pdfs:
        try:
            from pypdf import PdfReader
            for p in pdfs:
                try:
                    r = PdfReader(str(p))
                    texts.append((*_rel_name_dir(p),
                                  "\n".join(pg.extract_text() or "" for pg in r.pages)))
                except Exception as e:
                    print(f"  ⚠️ PDF 读取失败 {p.name}: {e}")
        except ImportError:
            print(f"  ⚠️ 有 {len(pdfs)} 个 PDF 但未装 pypdf，已跳过")
            print(f"     👉 pip install pypdf -i https://pypi.tuna.tsinghua.edu.cn/simple")

    # ④ docx —— 需 python-docx
    docxs = list(DOCS_DIR.rglob("*.docx"))
    if docxs:
        try:
            for p in docxs:
                try:
                    texts.append((*_rel_name_dir(p), read_docx(p)))
                except Exception as e:
                    print(f"  ⚠️ DOCX 读取失败 {p.name}: {e}")
        except ImportError:
            print(f"  ⚠️ 有 {len(docxs)} 个 DOCX 但未装 python-docx，已跳过")
            print(f"     👉 pip install python-docx -i https://pypi.tuna.tsinghua.edu.cn/simple")

    # ⑤ pptx —— 需 python-pptx
    pptxs = list(DOCS_DIR.rglob("*.pptx"))
    if pptxs:
        try:
            for p in pptxs:
                try:
                    texts.append((*_rel_name_dir(p), read_pptx(p)))
                except Exception as e:
                    print(f"  ⚠️ PPTX 读取失败 {p.name}: {e}")
        except ImportError:
            print(f"  ⚠️ 有 {len(pptxs)} 个 PPTX 但未装 python-pptx，已跳过")
            print(f"     👉 pip install python-pptx -i https://pypi.tuna.tsinghua.edu.cn/simple")
            print(f"     替代方案：PPT 另存为 .txt / .md，或用 WPS 批量导出文本")

    return texts


def report_missing(texts, all_files):
    """
    汇总「有哪些文档没被读进来」。

    为什么需要：缺包时脚本只是逐条 ⚠️ 打印，很容易被后面的
    向量化进度刷走，你只看最后「共 45 块」根本不知道少了什么。
    这里在最后做一次醒目汇总。
    """
    got = {t[1] for t in texts}
    missed = [p for p in all_files
              if p.is_file() and p.name not in got
              and p.suffix.lower() in SUPPORTED]
    if not missed:
        return

    print("\n" + "!" * 60)
    print(f"⚠️  有 {len(missed)} 个文档【没有被读进向量库】：")
    for p in missed:
        ext = p.suffix.lower()
        pkg = {".pdf": "pypdf", ".docx": "python-docx",
               ".pptx": "python-pptx"}.get(ext, "?")
        print(f"     · {p.name}  （缺 {pkg}）")

    pkgs = sorted({".pdf": "pypdf", ".docx": "python-docx",
                   ".pptx": "python-pptx"}[p.suffix.lower()]
                  for p in missed
                  if p.suffix.lower() in (".pdf", ".docx", ".pptx"))
    if pkgs:
        print(f"\n   一条命令装齐：")
        print(f"     pip install {' '.join(pkgs)} "
              f"-i https://pypi.tuna.tsinghua.edu.cn/simple")
        print(f"\n   装完必须重跑：python build_index.py")
        print(f"   （否则库里永远缺这些内容，检索会答非所问）")
    print("!" * 60 + "\n")


ALL_FILES = []          # load_docs() 填充，main() 最后用它做缺失汇总


# =========================================================
#  版本号自动递增
# =========================================================
def content_hash(chunks):
    """对切块结果取指纹，用来判断知识库内容是否真的变了"""
    import hashlib
    h = hashlib.md5()
    for c in chunks:
        h.update(c.encode("utf-8", errors="ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def _next_version(cur):
    """v1 → v2；非 vN 格式则加后缀"""
    import re as _re
    m = _re.match(r"^[vV](\d+)$", str(cur).strip())
    if m:
        return f"v{int(m.group(1)) + 1}"
    m2 = _re.search(r"(\d+)$", str(cur).strip())
    if m2:
        return f"{cur[:m2.start()]}{int(m2.group(1)) + 1}"
    return f"{cur}_2"


def _set_config_line(text, key, value):
    """精确替换 config.py 里某一行；不存在就追加到末尾"""
    import re as _re
    pat = _re.compile(rf"^(\s*){_re.escape(key)}\s*=\s*(.+?)\s*(#.*)?$",
                      _re.MULTILINE)
    new = f"{key} = {value}"
    if pat.search(text):
        return pat.sub(lambda m: new, text, count=1), True
    return text.rstrip() + f"\n\n{new}\n", False


def auto_bump_version(changed, old_hash, n_chunks) -> str:
    """
    建库内容有变 → 自动把 config.py 的 VERSION +1。

    为什么要按「内容是否变化」判断，而不是每次都 +1：
        重复建库（内容没变）也 +1 会让版本号无意义膨胀，
        对比表很快就会被一堆内容相同的版本塞满。

    返回是否真的递增了。
    """
    import shutil
    import time as _time
    from config import VERSION

    if not AUTO_BUMP_VERSION:
        print(f"\nℹ️ 自动递增已关闭（AUTO_BUMP_VERSION=False），"
              f"VERSION 保持 {VERSION}")
        return VERSION

    if FORCE_VERSION:
        # 手动指定：不管内容变没变都改
        return _write_version(FORCE_VERSION, VERSION)

    if not changed:
        if old_hash is None:
            print(f"\nℹ️ 首次建立该库，已记录指纹；VERSION 保持 {VERSION}")
        else:
            print(f"\nℹ️ 知识库内容与上次相同，VERSION 保持 {VERSION}")
        return VERSION

    if n_chunks == 0:
        print(f"\n⚠️ 块数为 0，不递增版本号")
        return VERSION

    return _write_version(_next_version(VERSION), VERSION)


def _write_version(new_ver, cur_ver):
    import shutil as _sh
    import time as _t
    cfg = Path(__file__).resolve().parent / "config.py"
    try:
        src = cfg.read_text(encoding="utf-8")
        backup = cfg.parent / f"config.py.bak.{_t.strftime('%m%d_%H%M%S')}"
        _sh.copy2(cfg, backup)
        src2, hit = _set_config_line(src, "VERSION", f'"{new_ver}"')
        cfg.write_text(src2, encoding="utf-8")
    except Exception as e:
        print(f"\n⚠️ 写版本号失败：{type(e).__name__}: {e}")
        print(f"   请手动把 config.py 的 VERSION 改成 \"{new_ver}\"")
        return cur_ver

    print("\n" + "=" * 60)
    print(f"🔢 VERSION 自动递增")
    print(f"     {cur_ver}  →  {new_ver}")
    print(f"   （config.py 已自动修改，原文件备份 → {backup.name}）")
    print("=" * 60)
    return new_ver


def print_next_steps(ver, bumped, changed):
    """告诉用户接下来该跑什么，避免漏步

    ver 必须显式传入：config.py 已被改写，但 Python 进程里的模块
    还是旧缓存，直接 from config import VERSION 会拿到旧值。
    """
    print("\n" + "-" * 60)
    print("【接下来】")
    if bumped:
        print(f"""  库变了，旧数据集已作废，必须重新生成：

    python run_batch.py          # 重新检索 + 生成（产出 eval_data_{ver}.xlsx）
    python make_reference.py     # 重新起草标准答案
    # 人工过一遍 data/eval_data_{ver}.xlsx 的 reference 列
    python run_eval.py           # 打分
    python compare.py            # 和上一版对比

  ⚠️ 不要跳过 run_batch.py：
     旧数据集里的 retrieved_contexts 来自旧库，和新库不可比。
""")
    elif changed is False:
        print(f"""  内容未变，数据集无需重建。
  若只想重跑评测：python run_eval.py
""")
    else:
        print(f"""    python run_batch.py
    python make_reference.py
    python run_eval.py
""")
    print("-" * 60)


# ---------- 2. 切块（中文友好，优先在句末断开）----------
def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            for sep in ("\n\n", "\n", "。", "！", "？", "；", ". "):
                idx = text.rfind(sep, start + size // 2, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


# ---------- 3. 向量化（带老版本 Ollama 兜底）----------
def embed(texts):
    try:
        r = requests.post(f"{OLLAMA_URL}/api/embed",
                          json={"model": EMBED_MODEL, "input": texts}, timeout=600)
        if r.status_code == 200:
            return r.json()["embeddings"]
    except Exception:
        pass

    out = []
    for t in texts:
        r = requests.post(f"{OLLAMA_URL}/api/embeddings",
                          json={"model": EMBED_MODEL, "prompt": t}, timeout=600)
        r.raise_for_status()
        out.append(r.json()["embedding"])
    return out


# ---------- 4. 主流程 ----------
def main():
    files = load_docs()
    if not files:
        print(f"❌ {DOCS_DIR} 里没有可用文档，请先放入 txt / md 文件")
        return

    chunks, metas = [], []
    empty_files = []
    for rel, name, dir_of, text in files:
        if not text or not text.strip():
            empty_files.append(name)
            print(f"  ⚠️ {name}: 提取到 0 字（文件可能是扫描件/纯图片/加密），已跳过")
            continue
        cs = split_text(text)
        for i, c in enumerate(cs):
            chunks.append(c)
            metas.append({"source": name, "doc_dir": dir_of,
                          "rel_path": rel, "chunk": i})
        print(f"  📄 {rel}: {len(text)} 字 → {len(cs)} 块")

    if empty_files:
        print(f"\n  💡 {len(empty_files)} 个文件没提取到文字，建议转成 txt 后放入 docs/")

    if not chunks:
        print("❌ 切块后为空，检查文档内容")
        return

    print(f"\n共 {len(chunks)} 块，开始向量化（{EMBED_MODEL}）...")

    vecs = []
    B = 32
    for i in range(0, len(chunks), B):
        vecs.extend(embed(chunks[i:i + B]))
        print(f"  {min(i + B, len(chunks))}/{len(chunks)}")

    arr = np.array(vecs, dtype="float32")
    # 归一化，之后检索直接点积就是余弦相似度
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    arr = arr / norm

    DB_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 内容指纹：用来判断「这次建库内容到底变没变」 ----------
    new_hash = content_hash(chunks)

    idx_path = DB_DIR / f"{COLLECTION}.json"
    old_hash = None
    old_existed = idx_path.exists()
    if old_existed:
        try:
            old_hash = json.loads(
                idx_path.read_text(encoding="utf-8")).get("content_hash")
        except Exception:
            old_hash = None
    if old_existed and old_hash is None:
        # 老版本脚本建的库没有指纹字段。此时无法判断内容是否变过，
        # 只能当「首次建立」处理 —— 所以第一次重跑不会自动递增 VERSION。
        print("\nℹ️ 现有索引是旧版脚本建的（没有 content_hash 指纹），")
        print("   本次会补写指纹；VERSION 不会自动 +1。")
        print("   要记录这次改动，请手动改 config.py 的 VERSION。")

    np.save(DB_DIR / f"{COLLECTION}.npy", arr)
    idx_path.write_text(
        json.dumps({"chunks": chunks, "metas": metas,
                    "embed_model": EMBED_MODEL,
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                    "content_hash": new_hash},
                   ensure_ascii=False),
        encoding="utf-8")

    print(f"\n✅ 建库完成 → {DB_DIR}")
    print(f"   collection = {COLLECTION}")
    print(f"   共 {len(chunks)} 块 | 维度 {arr.shape[1]} | "
          f"chunk={CHUNK_SIZE} overlap={CHUNK_OVERLAP}")

    # ---------- 版本号自动递增 ----------
    changed = (old_hash is not None and old_hash != new_hash)
    import config as _cfg
    ver_before = _cfg.VERSION
    final_ver = auto_bump_version(changed=changed, old_hash=old_hash,
                                  n_chunks=len(chunks))

    # ---------- 后续该跑什么 ----------
    print_next_steps(final_ver, bumped=(final_ver != ver_before),
                     changed=changed)

    # 最后再提醒一次哪些文档没读进来（放在最后才不会被进度刷走）
    if ALL_FILES:
        report_missing(files, ALL_FILES)


if __name__ == "__main__":
    main()
