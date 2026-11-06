# -*- coding: utf-8 -*-
"""
📊 参数与指标页（Streamlit）

这一页解决三件事：
  ① 一眼看清「当前到底在用什么参数」——包括哪些是从前端改来的
  ② 一眼看清「各版本 / 各次实验的指标」——读 data/metric_runs.jsonl 存档
  ③ 在前端改参数 → 跑实验 → 看指标变化（两种实验：快速检索 / 完整）

两种实验的差别（页面上也会说明）：
  快速检索实验：只重算 context_recall / context_precision，2~8 分钟
               （Ragas 这两个指标只依赖 问题+上下文+标准答案，跟回答无关）
  完整实验    ：后台跑 run_batch + make_reference + run_eval，40~60 分钟，四个指标全有
"""
import time

import streamlit as st

import config
import experiment
import metrics_store
import ui_noise  # noqa: F401  （导入即生效：压掉线程上下文警告）
import ui_state


# =========================================================
#  参数面板
# =========================================================
def _render_param_panel():
    st.markdown("### 🎛️ 参数")

    applied = getattr(config, "UI_OVERRIDES_APPLIED", {})
    if applied:
        st.info("当前 config 已被 `data/ui_overrides.json` 覆盖："
                + "、".join(f"`{k}`" for k in applied)
                + "　—　删掉这个文件或点下面的「恢复 config 默认」即可还原。")

    with st.form("param_form"):
        cols = st.columns(3)
        values = {}
        for i, spec in enumerate(ui_state.EDITABLE):
            col = cols[i % 3]
            key, kind = spec["key"], spec["kind"]
            cur = ui_state.get(key)
            with col:
                if kind == "int":
                    values[key] = st.number_input(
                        spec["label"], min_value=spec["min"],
                        max_value=spec["max"], value=int(cur),
                        step=1, help=spec["help"])
                elif kind == "float":
                    values[key] = st.number_input(
                        spec["label"], min_value=spec["min"],
                        max_value=spec["max"], value=float(cur),
                        step=spec["step"], help=spec["help"])
                elif kind == "bool":
                    values[key] = st.checkbox(spec["label"], value=bool(cur),
                                              help=spec["help"])
                else:
                    values[key] = st.text_input(spec["label"], value=str(cur),
                                                help=spec["help"])

        sync = st.checkbox(
            "同步到命令行脚本（写 data/ui_overrides.json）", value=True,
            help="勾上之后，python run_batch.py / run_eval.py 也会用这套参数；"
                 "不勾则只在当前界面生效。")
        c1, c2, c3 = st.columns([1, 1, 3])
        submitted = c1.form_submit_button("✅ 应用参数", type="primary")
        reset = c2.form_submit_button("↩️ 恢复 config 默认")

    if submitted:
        if values["RETRIEVE_K"] < values["TOP_K"]:
            st.warning(f"RETRIEVE_K({values['RETRIEVE_K']}) < TOP_K"
                       f"({values['TOP_K']})，已自动调大到 TOP_K 的值。")
            values["RETRIEVE_K"] = values["TOP_K"]
        ui_state.set_values(values)
        if sync:
            experiment.write_overrides(ui_state.snapshot())
            st.success("已应用，并写入 data/ui_overrides.json（命令行也生效）")
        else:
            st.success("已应用（仅当前界面；未同步给命令行）")
        # config 里的只读派生值不会变，但检索相关参数立即生效
        st.rerun()

    if reset:
        ui_state.reset_to_config()
        if experiment.clear_overrides():
            st.success("已恢复 config.py 默认值，并删除了 data/ui_overrides.json")
        else:
            st.success("已恢复 config.py 默认值")
        st.rerun()

    # ---- 只读参数 + 当前生效值 ----
    with st.expander("查看全部生效参数（含只读项）", expanded=True):
        eff = {}
        for key, label in ui_state.READONLY:
            eff[f"{label}（{key}）"] = getattr(config, key, "—")
        eff["GEN_MODEL"] = ui_state.get("GEN_MODEL")
        eff["JUDGE_MODEL"] = ui_state.get("JUDGE_MODEL")
        eff["TOP_K"] = ui_state.get("TOP_K")
        eff["RETRIEVE_K"] = ui_state.get("RETRIEVE_K")
        eff["RERANK_ENABLED"] = ui_state.get("RERANK_ENABLED")
        eff["HYBRID_ENABLED"] = ui_state.get("HYBRID_ENABLED")
        eff["TEMPERATURE"] = ui_state.get("TEMPERATURE")
        st.dataframe(
            {"参数": list(eff.keys()), "值": [str(v) for v in eff.values()]},
            hide_index=True, width="stretch")

        st.caption(
            "改 `EMBED_MODEL` / `CHUNK_SIZE` 需要重建向量库（python build_index.py），"
            "所以这里不提供热改；`NUM_CTX` 影响生成上下文窗口，改了要重启脚本。")


# =========================================================
#  指标面板
# =========================================================
def _metric_cards(metrics: dict, mode: str):
    order = (["context_recall", "context_precision"]
             if mode == "retrieval"
             else ["faithfulness", "answer_relevancy",
                   "context_recall", "context_precision"])
    cols = st.columns(len(order))
    for col, key in zip(cols, order):
        val = metrics.get(key)
        label = metrics_store.METRIC_LABELS.get(key, key)
        col.metric(label, "—" if val is None else f"{val:.3f}")


def _render_history():
    runs = metrics_store.load_all()
    st.markdown("### 📈 指标存档与对比")

    if not runs:
        st.info("还没有存档。下面「跑一次实验」或点「导入现有分数文件」即可开始积累。")
        return

    df = metrics_store.to_dataframe(runs)
    keep = ["时间", "模式", "版本", "TOP_K", "RETRIEVE_K", "RERANK_ENABLED",
            "GEN_MODEL", "faithfulness", "answer_relevancy",
            "context_recall", "context_precision", "样本数", "备注"]
    keep = [c for c in keep if c in df.columns]
    st.dataframe(df[keep], hide_index=True, width="stretch")

    # ---- 曲线：同一指标随时间的走向 ----
    metric_cols = [m for m in metrics_store.METRIC_NAMES if m in df.columns]
    if metric_cols and len(df) >= 2:
        chart = df[["时间"] + metric_cols].set_index("时间")
        st.markdown("**指标走势**")
        st.line_chart(chart, height=260)

    # ---- 两个一组做差值对比 ----
    if len(df) >= 2:
        st.markdown("**最近两次对比**")
        a, b = runs[-2], runs[-1]
        rows = []
        for key, label in metrics_store.METRIC_LABELS.items():
            va = (a.get("metrics") or {}).get(key)
            vb = (b.get("metrics") or {}).get(key)
            if va is None and vb is None:
                continue
            delta = None if (va is None or vb is None) else round(vb - va, 4)
            rows.append({
                "指标": label,
                f"上一次（{a.get('run_id')}）": va,
                f"本次（{b.get('run_id')}）": vb,
                "变化": delta,
                "判读": ("—" if delta is None else
                       ("🔺 变好" if delta > 0.02 else
                        ("🔻 变差" if delta < -0.02 else "➖ 噪声内"))),
            })
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption("本地/云端评委都有波动，|变化| < 0.05 基本可以当噪声。")

        # 参数差异
        pa, pb = a.get("params") or {}, b.get("params") or {}
        diff = {k: (pa.get(k), pb.get(k)) for k in set(pa) | set(pb)
                if pa.get(k) != pb.get(k)}
        if diff:
            st.markdown("**这两次之间改了什么参数**")
            st.dataframe(
                [{"参数": k, "上一次": v[0], "本次": v[1]}
                 for k, v in sorted(diff.items())],
                hide_index=True, width="stretch")
        else:
            st.caption("两次参数完全一致 —— 差异只来自评委/生成的随机性。")

    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("📤 导出 metric_runs.csv"):
        path = metrics_store.export_csv()
        st.success(f"已导出：{path.name}")
    if c2.button("🗑️ 清空存档", help="只删 data/metric_runs.jsonl，不动 xlsx/csv 结果"):
        metrics_store.clear()
        st.rerun()
    c3.caption(f"存档文件：{metrics_store.RUNS_FILE}")


# =========================================================
#  实验面板
# =========================================================
def _render_experiments():
    st.markdown("### 🧪 跑一次实验")

    tab_quick, tab_full = st.tabs(["⚡ 快速检索实验（2~8 分钟）",
                                   "🐢 完整实验（40~60 分钟，后台跑）"])

    # ---------- 快速 ----------
    with tab_quick:
        st.caption(
            "只重新检索并重算 `context_recall` / `context_precision`。"
            "Ragas 这两个指标只依赖「问题 + 检索到的上下文 + 标准答案」，"
            "跟模型写出来的回答无关，所以能省掉最慢的生成环节。"
            "**faithfulness / answer_relevancy 会留空**（回答没重新生成，算出来没意义）。")

        params = ui_state.snapshot()
        st.json({k: params[k] for k in ("TOP_K", "RETRIEVE_K",
                                        "RERANK_ENABLED", "GEN_MODEL",
                                        "JUDGE_MODEL")}, expanded=False)

        note = st.text_input("备注（写进存档，方便以后对比）",
                             value=f"TOP_K={params.get('TOP_K')}, "
                                   f"RETRIEVE_K={params.get('RETRIEVE_K')}, "
                                   f"rerank={params.get('RERANK_ENABLED')}",
                             key="quick_note")
        sync = st.checkbox("同时把这套参数写给命令行脚本", value=True,
                           key="quick_sync")

        if st.button("▶️ 开始快速检索实验", type="primary"):
            if sync:
                experiment.write_overrides(params)
            bar = st.progress(0.0, text="准备中…")

            def _progress(done, total, text):
                frac = 0.0 if not total else min(done / total, 1.0)
                bar.progress(frac, text=text[:80])

            t0 = time.perf_counter()
            try:
                with st.spinner("检索 + 评委打分中…"):
                    rec = experiment.run_retrieval_experiment(
                        params=params, progress=_progress, note=note)
                bar.progress(1.0, text="完成")
                st.success(f"完成，用时 {time.perf_counter() - t0:.0f}s，"
                           f"存档 run_id = {rec['run_id']}")
                _metric_cards(rec["metrics"], "retrieval")

                detail = rec.get("detail") or {}
                per_item = detail.get("per_item") or []
                if per_item:
                    st.markdown("**逐题得分**（找哪道题拉低了指标）")
                    st.dataframe(per_item, hide_index=True, width="stretch")
                st.caption(f"数据集：{detail.get('dataset')} · "
                           f"检索耗时 {detail.get('retrieval_seconds')}s")
            except Exception as e:                 # noqa: BLE001
                bar.empty()
                st.error(f"实验失败：{type(e).__name__}: {e}")
                st.caption("常见原因：数据集还没生成 / 还没有 reference / "
                           "评委模型不可用。先跑 python check_pipeline.py 看 ⑦。")

    # ---------- 完整 ----------
    with tab_full:
        st.caption(
            "后台执行 `run_batch.py` → `make_reference.py` → `run_eval.py`，"
            "四个指标全有。脚本用子进程跑，关掉页面也不会中断（但关掉 "
            "Streamlit 服务会中断）。")

        status = experiment.read_status()
        running = experiment.is_running()

        c1, c2, c3 = st.columns([1, 1, 2])
        skip_ref = c1.checkbox("跳过起草标准答案", value=False,
                               help="reference 已经人工填好时勾上，省一步")

        if c1.button("🧹 清空日志"):
            experiment.LOG_FILE.write_text("", encoding="utf-8")
            st.rerun()

        if running:
            st.warning(f"正在跑：{status.get('message')}（第 "
                       f"{status.get('step')}/{status.get('total')} 步）")
        else:
            if c2.button("🐢 启动完整实验", type="primary"):
                try:
                    experiment.start_full_experiment(
                        params=ui_state.snapshot(),
                        note=st.session_state.get("full_note", "前端完整实验"),
                        skip_reference=skip_ref)
                    st.rerun()
                except Exception as e:             # noqa: BLE001
                    st.error(f"启动失败：{type(e).__name__}: {e}")

        st.text_input("备注（写进指标存档）", key="full_note",
                      value=st.session_state.get("full_note", "前端完整实验"))

        if status.get("state") != "idle":
            st.markdown("**状态**")
            st.json(status, expanded=False)

        if status.get("state") == "done" and status.get("metrics"):
            st.success(f"上次完整实验完成：{status.get('run_id')}")
            _metric_cards(status["metrics"], "full")

        with st.expander("后台日志（最近 60 行）", expanded=running):
            st.code(experiment.read_log_tail(60) or "（还没有日志）",
                    language="text")

        if st.button("🔄 刷新状态"):
            st.rerun()

    # ---------- 补录 ----------
    with st.expander("📥 把已有的 scores_*.csv 补录进存档"):
        files = sorted(metrics_store.DATA_DIR.glob("scores_*.csv"))
        if not files:
            st.caption("data/ 下还没有 scores_*.csv")
        else:
            pick = st.selectbox("选择分数文件", [p.name for p in files])
            note = st.text_input("备注", value="从命令行 run_eval.py 的结果补录",
                                 key="import_note")
            if st.button("补录"):
                path = metrics_store.DATA_DIR / pick
                params = experiment.default_params()
                rec = metrics_store.import_scores_csv(path, params=params,
                                                      note=note)
                if rec:
                    st.success(f"已补录 {rec['run_id']}：{rec['metrics']}")
                    st.rerun()
                else:
                    st.error("这个文件里没有可识别的指标列")


# =========================================================
#  页面入口
# =========================================================
def render():
    st.subheader("📊 参数与指标")
    st.caption("改参数 → 跑实验 → 看指标变化。所有实验都会写进 "
               f"`{metrics_store.RUNS_FILE.name}`，可随时对比。")

    latest = metrics_store.latest()
    if latest:
        st.markdown(f"**最近一次实验**：{latest.get('run_id')}"
                    f"（{latest.get('ts')} · 模式 {latest.get('mode')} · "
                    f"样本 {latest.get('n_samples')}）")
        _metric_cards(latest.get("metrics") or {}, latest.get("mode") or "full")
    else:
        st.info("还没有实验记录。先在下面「导入现有分数文件」，或直接跑一次实验。")

    st.divider()
    _render_param_panel()
    st.divider()
    _render_experiments()
    st.divider()
    _render_history()

# ---- 页面入口 ----
# 文件式页面（st.Page("page_xxx.py")）会从上到下执行，所以这里直接调用。
render()
