# -*- coding: utf-8 -*-
"""
📚 知识库 RAG 控制台 —— Streamlit 入口

启动：
    streamlit run app.py

两个页面（真正的页面文件，可以直接深链）：
    page_chat.py      💬 问答            —— 流式问答 + 引用原文 + 耗时
    page_metrics.py   📊 参数与指标       —— 看参数 / 看指标存档 / 改参数跑实验

（旧版是一个单页大文件，改参数只能改 config.py 再重启、看不到历史指标；
  现在是文件式多页 + 前端调参 + 指标存档。）
"""
import streamlit as st

st.set_page_config(page_title="知识库 RAG 控制台", page_icon="📚",
                   layout="wide")

# 控制台降噪：压掉「Thread 'xxx': missing ScriptRunContext」那条日志
import ui_noise  # noqa: F401  （导入即生效）

# 参数状态要在页面渲染前初始化（两个页面共用同一份）
import ui_state

ui_state.init()

nav = st.navigation([
    st.Page("page_chat.py", title="问答", icon="💬", default=True),
    st.Page("page_metrics.py", title="参数与指标", icon="📊"),
])

nav.run()
