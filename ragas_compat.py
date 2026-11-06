# -*- coding: utf-8 -*-
"""
ragas 导入兼容补丁

背景（一句话）：
    某些第三方库在 import 阶段会引用「装了，但那个子模块已经被删掉」的模块，
    于是 `import ragas` 直接抛 ModuleNotFoundError，打分脚本就跑不起来。

做法（不写死任何第三方包名，自动适配）：
    尝试导入 ragas；如果它因为某个【子模块】缺失而失败，就现场造一个
    空占位模块塞进 sys.modules 再重试。
    判定条件很保守：只有「顶层包确实已安装、只是这个子模块不存在」才补；
    如果连顶层包都没装，说明是真缺依赖，照常报错，不会掩盖问题。

用法（必须早于任何 ragas import）：
    import ragas_compat
    ragas_compat.install()
"""
import importlib
import importlib.util
import sys
import types

_MAX_RETRY = 6          # 最多补几个模块（正常只会补 1~2 个，留点余量）

_installed = False


def _top_pkg_installed(module_name: str) -> bool:
    top = module_name.split(".", 1)[0]
    try:
        return importlib.util.find_spec(top) is not None
    except (ImportError, ValueError):
        return False


def _stub(module_name: str) -> None:
    """把一个缺失的子模块替换成空占位（属性按需动态生成）"""
    mod = types.ModuleType(module_name)
    mod.__doc__ = "占位模块：原模块已从依赖中移除"

    def _missing(attr):                       # 访问任何属性都拿到空类
        return type(attr, (object,), {})

    mod.__getattr__ = _missing                # type: ignore[attr-defined]
    sys.modules[module_name] = mod


def install():
    """
    幂等：重复调用无副作用。

    会真正 import 一次 ragas（约 1~2 秒），因为补位必须发生在导入过程中。
    """
    global _installed
    if _installed:
        return

    tried = set()
    for _ in range(_MAX_RETRY):
        try:
            importlib.import_module("ragas")
            break
        except ModuleNotFoundError as e:
            name = getattr(e, "name", None) or ""
            # 只处理「子模块」缺失；顶层包缺失说明真的没装，直接抛出
            if "." not in name or name in tried:
                raise
            if not _top_pkg_installed(name):
                raise
            tried.add(name)
            _stub(name)
    else:                                     # 补了 _MAX_RETRY 次仍失败
        importlib.import_module("ragas")      # 让它把真实错误抛出来

    _installed = True


def check():
    """返回 (是否可用, 问题列表)。给 check_pipeline.py 用。"""
    problems = []
    try:
        install()
    except Exception as e:                    # noqa: BLE001
        problems.append(f"ragas: {type(e).__name__}: {e}")
    for pkg in ("openai",):
        try:
            importlib.import_module(pkg)
        except Exception as e:                # noqa: BLE001
            problems.append(f"{pkg}: {type(e).__name__}: {e}")
    return (len(problems) == 0), problems
