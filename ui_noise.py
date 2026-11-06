# -*- coding: utf-8 -*-
"""
🔇 控制台降噪

Streamlit 有一个特点：只要在「非脚本线程」里碰了 `st.session_state`、
`st.cache_resource` 这类 API，它就会打印一条：

    Thread 'xxx-worker': missing ScriptRunContext! This warning can be
    ignored when running in bare mode.

我们的检索/生成放在子线程里跑（为了做超时保护），所以正确的做法是
**把所有 Streamlit 调用都留在主线程**（参数在主线程读好再传进去）。
即使这样，某些情况下仍会漏出一条，而这条日志对使用者没有任何价值，
所以额外把它的级别压到 ERROR 作为兜底。

注意：这不是「屏蔽所有警告」，只针对这一个 logger。
"""
import logging

_LOGGER_NAME = "streamlit.runtime.scriptrunner_utils.script_run_context"
_done = False


def silence():
    """幂等：把 ScriptRunContext 那条警告压掉"""
    global _done
    if _done:
        return
    logging.getLogger(_LOGGER_NAME).setLevel(logging.ERROR)
    _done = True


# 导入即生效（各页面/入口只要 import 就行）
silence()
