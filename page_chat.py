# -*- coding: utf-8 -*-
"""
💬 问答页（Streamlit）

内容：
  · 顶部一行实时参数徽标（改完参数立刻反映）
  · 流式回答 + 三段耗时（检索 / 首字 / 总时长）+ 生成速度
  · 引用原文：先给一张「来源 / 分数 / 字数」汇总表，再逐段展开看原文（关键词高亮）

说明：
  检索走 retrieval.Retriever（和 CLI、实验引擎共用一套逻辑），
  参数来自 ui_state（前端覆盖优先，其次 config.py）。
"""
import html
import re
import time

import streamlit as st

import config
import ui_noise  # noqa: F401  （导入即生效：压掉线程上下文警告）
import ui_state
from retrieval import Retriever, doc_dir_stats, load_index, meta_dir
from rerank import Reranker

HL_COLOR = "#FFE066"          # 关键词高亮色

SEARCH_TIMEOUT = int(getattr(config, "SEARCH_TIMEOUT", 60) or 60)
GEN_TIMEOUT = int(getattr(config, "GEN_TIMEOUT", 180) or 180)


# =========================================================
#  缓存：向量库只读一次、rerank 模型只加载一次
# =========================================================
@st.cache_resource(show_spinner="加载向量库…")
def _load_index_cached():
    return load_index()


@st.cache_resource(show_spinner="加载 rerank 模型…")
def _reranker_cached():
    return Reranker()


def _make_retriever(doc_dirs=None) -> Retriever:
    use_rerank = bool(ui_state.get("RERANK_ENABLED", False))
    return Retriever(
        top_k=int(ui_state.get("TOP_K", 5)),
        pool_k=int(ui_state.get("RETRIEVE_K", 20)),
        use_rerank=use_rerank,
        reranker=_reranker_cached() if use_rerank else None,
        index=_load_index_cached(),
        doc_dirs=doc_dirs or [],
        use_hybrid=bool(ui_state.get("HYBRID_ENABLED", False)),
    )


# =========================================================
#  文档范围筛选（按 docs/ 下的文件夹）
# =========================================================
SCOPE_KEY = "doc_scope_labels"      # 存的是界面上的显示标签


def _dir_options():
    """
    返回 [(显示标签, 真实文件夹名, 文档数, 块数)]。

    根目录显示成「（根目录）」；无法判定归属的 "?" 不放进选项
    （它只会出现在"全部"里）。
    """
    try:
        stats = doc_dir_stats(_load_index_cached())
    except Exception:                              # noqa: BLE001
        return []

    opts = []
    for d in sorted(stats, key=lambda x: (x in ("", "?"), x)):
        if d == "?":
            continue
        label = "（根目录）" if d == "" else d
        opts.append((label, d, stats[d]["docs"], stats[d]["chunks"]))
    return opts


def _doc_scope_selector():
    """渲染文件夹多选 + 全选/清空，返回选中的真实文件夹名列表（空=全部）"""
    opts = _dir_options()
    if not opts:
        return []

    labels = [o[0] for o in opts]
    label_to_dir = {o[0]: o[1] for o in opts}

    # 索引变化后可能出现失效的旧选择，先清理
    cur = [x for x in st.session_state.get(SCOPE_KEY, []) if x in label_to_dir]
    st.session_state[SCOPE_KEY] = cur

    def _select_all():
        st.session_state[SCOPE_KEY] = list(labels)

    def _clear_all():
        st.session_state[SCOPE_KEY] = []

    c1, c2, c3 = st.columns([6, 1, 1], vertical_alignment="bottom", gap="small")
    with c1:
        chosen = st.multiselect(
            "文档范围", options=labels, key=SCOPE_KEY,
            placeholder="全部文档（不筛选）",
            help="只在这些文件夹里检索。可以选一个、选多个，或者点「全选」。"
                 "清空 = 不筛选。",
            label_visibility="collapsed")
    with c2:
        st.button("全选", on_click=_select_all, width="stretch",
                  help="选中全部文件夹（等同不筛选）")
    with c3:
        st.button("清空", on_click=_clear_all, width="stretch",
                  help="取消筛选，回到全部文档")

    selected_dirs = [label_to_dir[x] for x in chosen if x in label_to_dir]

    # 一行说明当前范围里到底有多少内容
    by_label = {o[0]: o for o in opts}
    if chosen:
        docs = sum(by_label[x][2] for x in chosen if x in by_label)
        chunks = sum(by_label[x][3] for x in chosen if x in by_label)
        st.caption(f"🔎 当前只在 **{len(chosen)}** 个文件夹里检索："
                   f"{'、'.join(chosen)} —— 共 {docs} 份文档 / {chunks} 块")
    else:
        all_docs = sum(o[2] for o in opts)
        all_chunks = sum(o[3] for o in opts)
        st.caption(f"🔎 当前范围：全部文档 —— {all_docs} 份 / {all_chunks} 块"
                   f"（{len(opts)} 个文件夹）")

    return selected_dirs


# =========================================================
#  超时保护
# =========================================================
def _stream_with_timeout(gen_factory, timeout, on_first=None, on_piece=None):
    """在子线程里跑流式生成，首字超时后掐断，避免界面永久转圈"""
    import queue as _queue
    import threading

    q: _queue.Queue = _queue.Queue()
    state = {"first": False, "t0": time.perf_counter(), "err": None}
    acc = []

    def worker():
        try:
            for piece in gen_factory():
                q.put(piece)
        except BaseException as e:                # noqa: BLE001
            state["err"] = f"{type(e).__name__}: {e}"
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True, name="gen-worker").start()

    while True:
        try:
            piece = q.get(timeout=timeout if not state["first"] else 30)
        except Exception:                          # noqa: BLE001
            if not state["first"]:
                state["err"] = (f"等待首字超过 {timeout}s —— "
                                f"模型可能正在加载（首次调用）或服务不可用")
            break
        if piece is None:
            break
        if not state["first"]:
            state["first"] = True
            if on_first:
                on_first((time.perf_counter() - state["t0"]) * 1000)
        acc.append(piece)
        if on_piece:
            on_piece("".join(acc))

    if state["err"] is None and not acc:
        state["err"] = "模型没有返回任何内容"
    return "".join(acc), state["err"]


def _run_with_timeout(fn, timeout, label="操作"):
    import threading

    box = {}

    def target():
        try:
            box["v"] = fn()
        except BaseException as e:                 # noqa: BLE001
            box["e"] = e

    t = threading.Thread(target=target, daemon=True, name=f"{label}-worker")
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, f"{label}超过 {timeout}s 未完成"
    if "e" in box:
        return None, f"{type(box['e']).__name__}: {box['e']}"
    return box.get("v"), None


# =========================================================
#  生成
# =========================================================
def strip_think(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def generate_stream(contexts, question, gen_model=None, temperature=None):
    """
    流式生成。

    ⚠️ 参数必须由调用方（主线程）传进来，不要在这里读 ui_state / st.session_state：
       这个函数跑在 gen-worker 子线程里，而 Streamlit 的 session_state 只能在
       主线程访问。在子线程里读它虽然通常仍能工作，但每读一次就会打印一条
       「Thread 'gen-worker': missing ScriptRunContext!」，把控制台刷满。
    """
    import requests

    gen_model = gen_model or config.GEN_MODEL
    temperature = 0.0 if temperature is None else float(temperature)

    ctx = "\n\n---\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    user = f"【参考资料】\n{ctx}\n\n【问题】\n{question}"
    if getattr(config, "DISABLE_THINK", True):
        user += "\n/no_think"

    payload = {
        "model": gen_model,
        "messages": [
            {"role": "system", "content": config.RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        "stream": True,
        "options": {"temperature": temperature,
                    "num_ctx": int(config.NUM_CTX)},
    }
    r = requests.post(f"{config.OLLAMA_URL}/api/chat", json=payload,
                      stream=True, timeout=900)
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        import json
        try:
            data = json.loads(line.decode("utf-8"))
        except Exception:                          # noqa: BLE001
            continue
        if data.get("done"):
            break
        chunk = (data.get("message") or {}).get("content", "")
        if chunk:
            yield chunk


# =========================================================
#  关键词高亮（不装 jieba：在问题和原文之间找最长公共子串）
# =========================================================
STOPWORDS = set("""
的 了 是 在 有 和 就 不 都 一 一个 上 也 很 到 说 要 去 会 着 没有 看 好 自己
这 那 什么 怎么 如何 哪些 请 想 问 一下 吗 呢 吧 啊 可以 能否 以及 或者 与 及
对 对于 关于 中 下 里 后 前 我们 你们 他们 它 这个 那个 因为 所以 但是 如果
""".split())

CJK = r"\u4e00-\u9fff"


def extract_keywords(question, contexts, min_len=2, max_len=12):
    import re as _re

    q = _re.sub(r"[^\w" + CJK + "]+", "", question.strip())
    if not q or not contexts:
        return []
    joined = "\n".join(contexts)
    found = set()
    for i in range(len(q)):
        for j in range(min(len(q), i + max_len), i + min_len - 1, -1):
            sub = q[i:j]
            if len(sub) < min_len:
                break
            if sub in joined:
                found.add(sub)
                break

    found = {w for w in found
             if w not in STOPWORDS and not w.isdigit() and len(w) >= min_len}
    ordered = sorted(found, key=len, reverse=True)
    keep = []
    for w in ordered:
        if not any(w in k for k in keep):
            keep.append(w)
    return keep[:20]


def highlight(text, keywords):
    esc = html.escape(text)
    for w in sorted(keywords, key=len, reverse=True):
        ew = html.escape(w)
        esc = esc.replace(ew, f"<mark style='background:{HL_COLOR};"
                              f"padding:0 2px;border-radius:2px'>{ew}</mark>")
    return esc


# =========================================================
#  界面
# =========================================================
def _param_badges():
    """当前生效参数：一排小 chip（用半透明灰，浅色/深色主题都看得清）"""
    items = [
        ("生成", ui_state.get("GEN_MODEL")),
        ("TOP_K", ui_state.get("TOP_K")),
        ("候选", ui_state.get("RETRIEVE_K")),
        ("混合", "BM25+向量" if ui_state.get("HYBRID_ENABLED") else "关"),
        ("rerank", "开" if ui_state.get("RERANK_ENABLED") else "关"),
        ("温度", ui_state.get("TEMPERATURE")),
        ("向量库", config.COLLECTION),
    ]
    chips = "".join(
        "<span style='display:inline-block;padding:.14rem .55rem;margin:0 .3rem .3rem 0;"
        "border:1px solid rgba(128,128,128,.3);border-radius:999px;font-size:.78rem;"
        f"background:rgba(128,128,128,.08);'>{html.escape(str(k))}&nbsp;"
        f"<b>{html.escape(str(v))}</b></span>"
        for k, v in items)
    st.markdown(f"<div style='margin:.1rem 0 .5rem 0'>{chips}</div>",
                unsafe_allow_html=True)


# =========================================================
#  首屏空状态（没有对话时显示，别让页面空着）
# =========================================================
def _corpus_stats():
    """返回 (文档数, 片段数)，读不到库时返回 (0, 0)"""
    try:
        _, chunks, metas = _load_index_cached()
        sources = {(m or {}).get("source", "?") for m in metas}
        return len(sources), len(chunks)
    except Exception:                              # noqa: BLE001
        return 0, 0


def _suggestion_questions(limit=4):
    """从 questions.txt 里取几道题做示例（太长的不放按钮上）"""
    try:
        lines = [ln.strip() for ln in
                 config.QUESTIONS.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
    except Exception:                              # noqa: BLE001
        lines = []
    out = [q for q in lines if len(q) <= 30][:limit]
    return out or lines[:limit]


def _empty_state():
    """居中标语 + 语料规模 + 示例问题 + 能力卡片；返回被点击的示例问题"""
    n_docs, n_chunks = _corpus_stats()
    rerank_on = bool(ui_state.get("RERANK_ENABLED", False))
    gen = ui_state.get("GEN_MODEL", config.GEN_MODEL)

    if n_chunks == 0:
        st.warning("还没有向量库：先运行 `python build_index.py` 再提问。")

    st.markdown(
        "<div style='text-align:center;padding:2.2rem 0 .2rem 0'>"
        "<div style='font-size:3.2rem;line-height:1'>📚</div>"
        "<h1 style='font-size:2.15rem;font-weight:800;margin:.75rem 0 .55rem 0'>"
        "让知识，可被召回</h1>"
        "<p style='font-size:1rem;margin:0 auto;max-width:34rem;line-height:1.75;"
        "opacity:.72'>从你的文档里检索、再回答 —— 每句结论都能点开看出处；"
        "资料里没有的内容，会明确说无法回答，而不是编。</p>"
        f"<div style='margin-top:1rem;font-size:.85rem;opacity:.55'>"
        f"{n_docs} 份文档 · {n_chunks} 个知识片段 · {html.escape(str(gen))} · "
        f"rerank {'开' if rerank_on else '关'}</div>"
        "</div>",
        unsafe_allow_html=True)

    picked = None
    suggestions = _suggestion_questions()
    if suggestions:
        st.markdown(
            "<div style='text-align:center;font-size:.9rem;opacity:.65;"
            "margin:1.5rem 0 .5rem 0'>试试这些问题</div>",
            unsafe_allow_html=True)
        _, mid, _ = st.columns([1, 3.2, 1], gap="small")
        with mid:
            cols = st.columns(2, gap="small")
            for i, s in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(s, key=f"hero_sug_{i}", width="stretch"):
                        picked = s

    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 6, 1], gap="small")
    with mid:
        c1, c2, c3 = st.columns(3, gap="medium")
        cards = [
            ("📄", "带出处的回答", "引用片段可展开看原文，关键词高亮。"),
            ("🎯", "先检索，再回答", "粗排 + rerank 精排，信息不足不硬编。"),
            ("📊", "可复现的指标", "「参数与指标」页改参数即可跑实验并存档。"),
        ]
        for col, (icon, title, desc) in zip((c1, c2, c3), cards):
            with col:
                st.markdown(f"**{icon} {title}**")
                st.caption(desc)

    return picked


def _render_sources(docs, metas, scores, kws, timings, note, expanded=False,
                    hybrid=False):
    rows = []
    for i, (d, mt, sc) in enumerate(zip(docs, metas, scores), 1):
        src = mt.get("source", "?") if isinstance(mt, dict) else str(mt)
        folder = meta_dir(mt) if isinstance(mt, dict) else ""
        rows.append({"#": i, "文件夹": folder or "（根目录）", "来源": src,
                     "分数": round(float(sc), 4), "字数": len(d)})
    st.markdown("**引用原文汇总**")
    try:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    except Exception:                              # noqa: BLE001
        st.table(rows)

    st.caption(f"召回：{'混合（BM25+向量）' if hybrid else '向量'} · "
               f"rerank：{note} · 检索 {timings['search_ms']:.0f} ms · "
               f"首字 {timings.get('first_ms', 0):.0f} ms · "
               f"总 {timings['total_s']:.1f} s · "
               f"生成 {timings.get('chars_per_s', 0):.1f} 字/秒")

    for i, (d, mt, sc) in enumerate(zip(docs, metas, scores), 1):
        src = mt.get("source", "?") if isinstance(mt, dict) else str(mt)
        folder = meta_dir(mt) if isinstance(mt, dict) else ""
        label = f"{folder}/{src}" if folder else src
        with st.expander(f"[{i}] {label} · 分数 {float(sc):.3f} · {len(d)} 字",
                         expanded=expanded):
            st.markdown(highlight(d, kws), unsafe_allow_html=True)


def render():
    st.subheader("💬 知识库问答")
    _param_badges()
    scope = _doc_scope_selector()          # 选中的文件夹（空 = 全部文档）

    if "msgs" not in st.session_state:
        st.session_state["msgs"] = []

    # 没有对话时显示首屏空状态（居中标语 + 示例问题 + 能力卡片）
    suggested_q = None
    if not st.session_state["msgs"]:
        suggested_q = _empty_state()
    else:
        if st.button("🧹 清空对话", type="secondary"):
            st.session_state["msgs"] = []
            st.rerun()

    # ---- 历史消息 ----
    for m in st.session_state["msgs"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"], unsafe_allow_html=True)
            if m.get("sources"):
                with st.expander(f"📄 引用原文（{len(m['sources'])} 段）"):
                    for i, s in enumerate(m["sources"], 1):
                        st.caption(f"[{i}] {s['source']} · 分数 {s['score']:.3f}")
                        st.markdown(s["html"], unsafe_allow_html=True)
                        st.divider()

    q = st.chat_input("输入你的问题，回车发送")
    if not q:
        q = suggested_q          # 点了示例问题就当作输入
    if not q:
        return

    with st.chat_message("user"):
        st.markdown(html.escape(q))

    with st.chat_message("assistant"):
        st_prog = st.empty()
        scope_txt = f"（限定 {'、'.join(scope)}）" if scope else ""
        st_prog.caption(f"① 正在检索…{scope_txt}（查询向量 + 向量比对 + rerank）")

        # ⚠️ Retriever 必须先在【主线程】构造好再丢进子线程：
        #    _make_retriever 会碰 @st.cache_resource，子线程里碰它就会刷
        #    「missing ScriptRunContext」警告。
        try:
            retriever = _make_retriever(scope)
        except Exception as e:                       # noqa: BLE001
            st_prog.empty()
            st.error(f"初始化检索器失败：{type(e).__name__}: {e}")
            return

        res, err = _run_with_timeout(
            lambda: retriever.search(q), SEARCH_TIMEOUT, "检索")
        if err:
            st_prog.empty()
            st.error(f"检索失败：{err}")
            st.caption("排查：\n"
                       f"  1. Ollama 在跑吗：`curl {config.OLLAMA_URL}/api/tags`\n"
                       f"  2. 嵌入模型装了吗：`ollama list | findstr "
                       f"{config.EMBED_MODEL}`\n"
                       f"  3. 文档范围是不是选得太窄了？点上方的「清空」取消筛选\n"
                       "  4. 完整体检：`python check_pipeline.py`")
            return
        st_prog.empty()

        if not res.docs:
            st.warning("没有检索到任何内容")
            return

        kws = extract_keywords(q, res.docs)

        # ---- 生成（流式，带首字超时）----
        # 参数在主线程读好，再传进子线程（子线程里读 session_state 会刷警告）
        gen_model = ui_state.get("GEN_MODEL", config.GEN_MODEL)
        gen_temp = float(ui_state.get("TEMPERATURE", 0.0))

        ph = st.empty()
        ph.caption("② 模型生成中…（首次调用需加载模型，可能几十秒）")
        t_gen0 = time.perf_counter()
        first_ms_box = {}

        def _on_first(ms):
            first_ms_box["v"] = ms
            ph.caption(f"③ 生成中…（首字 {ms:.0f} ms）")

        acc, err_gen = _stream_with_timeout(
            lambda: generate_stream(res.docs, q, gen_model, gen_temp),
            GEN_TIMEOUT,
            on_first=_on_first,
            on_piece=lambda text: ph.markdown(text),
        )
        gen_s = time.perf_counter() - t_gen0

        if err_gen:
            ph.empty()
            st.error(f"生成失败：{err_gen}")
            st.caption("排查：\n"
                       f"  1. 首次调用？先预热：`ollama run "
                       f"{ui_state.get('GEN_MODEL')} \"你好\"`\n"
                       f"  2. 想更快：把 GEN_MODEL 换成 qwen3:4b\n"
                       f"  3. 提高上限：GEN_TIMEOUT（当前 {GEN_TIMEOUT}s）")
            return

        acc = strip_think(acc)
        ph.markdown(acc)

        timings = {
            "search_ms": res.elapsed_ms,
            "total_s": gen_s,
            "chars_per_s": len(acc) / gen_s if gen_s > 0 else 0,
            "first_ms": first_ms_box.get("v", 0.0),
        }

        st.divider()
        _render_sources(res.docs, res.metas, res.scores, kws, timings,
                        res.note, expanded=False, hybrid=res.hybrid)

        sources = [{"source": (mt.get("source", "?") if isinstance(mt, dict)
                               else str(mt)),
                    "score": float(sc),
                    "html": highlight(d, kws)}
                   for d, mt, sc in zip(res.docs, res.metas, res.scores)]

        st.session_state["msgs"].append({"role": "user",
                                         "content": html.escape(q)})
        st.session_state["msgs"].append({"role": "assistant", "content": acc,
                                         "sources": sources})

# ---- 页面入口 ----
# 文件式页面（st.Page("page_xxx.py")）会从上到下执行，所以这里直接调用。
render()
