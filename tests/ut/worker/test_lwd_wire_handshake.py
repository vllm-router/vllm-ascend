#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""纯 CPU 用例：``lwd_wire._lwd_wait_with_timeout`` 的三条路径。

``lwd_wire`` 顶层 import torch/NPU，本机 UT 环境未必装有这些依赖；沿用
``test_lwd_landed_intervals`` 的 AST 提取手法：只把被测函数抠出来 exec，
不 import 整个模块。
"""

import ast
import os
import threading
import time
import unittest

_WIRE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "vllm_ascend",
        "distributed",
        "lwd_wire.py",
    )
)


def _load_function(name: str):
    with open(_WIRE, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            segment = ast.get_source_segment(src, node)
            if segment is None:
                break
            namespace = {"threading": threading}
            exec(segment, namespace)  # noqa: S102 - 仅用于单元测试
            return namespace[name]
    raise AssertionError(f"lwd_wire.py 中找不到函数 {name}")


class _ImmediateWork:
    def wait(self) -> None:
        return None


class _RaisingWork:
    def wait(self) -> None:
        raise RuntimeError("boom")


class _NeverWork:
    """永不完成；测试结束时 unblock() 放行守护线程。"""

    def __init__(self) -> None:
        self._release = threading.Event()

    def wait(self) -> None:
        self._release.wait()

    def unblock(self) -> None:
        self._release.set()


class _SlowWork:
    def __init__(self, delay: float) -> None:
        self._delay = delay

    def wait(self) -> None:
        time.sleep(self._delay)


class TestWaitWithTimeout(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # staticmethod: 避免函数作为类属性被绑定成方法(会多收一个 self)
        cls.wait_fn = staticmethod(_load_function("_lwd_wait_with_timeout"))

    def test_returns_when_work_completes(self) -> None:
        self.wait_fn(_ImmediateWork(), 1.0, "engine=0 dir=DOWN")

    def test_raises_timeout(self) -> None:
        work = _NeverWork()
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError) as ctx:
                self.wait_fn(work, 0.05, "engine=0 dir=DOWN peer=9 rank=0")
        finally:
            work.unblock()
        elapsed = time.monotonic() - started
        self.assertIn("timeout", str(ctx.exception))
        self.assertIn("engine=0 dir=DOWN", str(ctx.exception))
        self.assertGreaterEqual(elapsed, 0.04)
        self.assertLess(elapsed, 1.0)

    def test_reraises_work_exception(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self.wait_fn(_RaisingWork(), 1.0, "engine=1 dir=UP")
        self.assertIn("failed", str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))

    def test_slow_work_within_timeout_passes(self) -> None:
        self.wait_fn(_SlowWork(0.05), 1.0, "engine=0 dir=UP")

    def test_timeout_message_has_context(self) -> None:
        work = _NeverWork()
        try:
            with self.assertRaises(TimeoutError) as ctx:
                self.wait_fn(work, 0.02, "engine=2 dir=DOWN-send peer=7")
        finally:
            work.unblock()
        message = str(ctx.exception)
        self.assertIn("engine=2", message)
        self.assertIn("peer=7", message)


if __name__ == "__main__":
    unittest.main()
