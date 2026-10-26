# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""``make_recv_hint``/``hint_fields`` 的 refills 编解码回环测试(§5.3)。

refills = 与 req_ids 平行的逐请求 refill 标志(``LwdRangeItem.is_refill``
透传):缺省/短表 = 全 False(兼容旧版引擎),长表 = 版本错配判死。

torch-free:``pre_recv`` 自身只依赖 ``vllm.logger`` 与
``vllm_ascend.distributed.lwd_comm.types``,两者都按文件位置加载/打桩,
本测试在无 torch/NPU 的机器上直接跑。
"""

import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
TYPES_MOD = ROOT / "vllm_ascend/distributed/lwd_comm/types.py"
MODULE = ROOT / "vllm_ascend/distributed/lwd_comm/pre_recv.py"

# pre_recv 只从 vllm.logger 取 init_logger;桩掉整包,避免拉起
# torch 依赖链(与 tests/ut/worker/test_lwd_host_stage.py 同款手法)。
if "vllm.logger" not in sys.modules:
    fake_vllm = types.ModuleType("vllm")
    fake_logger_mod = types.ModuleType("vllm.logger")
    fake_logger_mod.init_logger = logging.getLogger
    fake_vllm.logger = fake_logger_mod
    sys.modules.setdefault("vllm", fake_vllm)
    sys.modules["vllm.logger"] = fake_logger_mod


def load_from(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# types.py 是自包含的(enum/dataclass,无第三方依赖);为
# ``from vllm_ascend.distributed.lwd_comm.types import ...`` 补齐包链。
for pkg in ("vllm_ascend", "vllm_ascend.distributed",
            "vllm_ascend.distributed.lwd_comm"):
    if pkg not in sys.modules:
        stub = types.ModuleType(pkg)
        stub.__path__ = []
        sys.modules[pkg] = stub
load_from(TYPES_MOD, "vllm_ascend.distributed.lwd_comm.types")
module = load_from(MODULE, "lwd_pre_recv")

make_recv_hint = module.make_recv_hint
hint_fields = module.hint_fields


class TestRecvHintRefills(unittest.TestCase):
    def make_hint(self, **kw):
        args = dict(
            seqno=7,
            num_tokens=5,
            req_ids=["a", "b"],
            rows=[2, 3],
            offsets=[0, 2],
            prompt_offsets=[4, 0],
            prompt_lens=[10, 8],
        )
        args.update(kw)
        return make_recv_hint(**args)

    def test_roundtrip_with_refills(self):
        hint = self.make_hint(refills=[False, True])
        self.assertEqual(hint["refills"], [False, True])
        fields = hint_fields(hint)
        self.assertIsNotNone(fields)
        # 既有 8 个字段的位置不动,refills 追加在末位
        self.assertEqual(fields[0], 7)
        self.assertEqual(fields[3], ["a", "b"])
        self.assertEqual(fields[6], [4, 0])
        self.assertEqual(fields[7], [10, 8])
        self.assertEqual(fields[8], [False, True])

    def test_missing_refills_defaults_all_false(self):
        fields = hint_fields(self.make_hint())
        self.assertIsNotNone(fields)
        self.assertEqual(fields[8], [False, False])

    def test_short_refills_padded_false(self):
        hint = self.make_hint(refills=[True])
        fields = hint_fields(hint)
        self.assertIsNotNone(fields)
        self.assertEqual(fields[8], [True, False])

    def test_long_refills_rejected(self):
        hint = self.make_hint(refills=[True, False, True])
        self.assertIsNone(hint_fields(hint))

    def test_refill_values_coerced_to_bool(self):
        hint = self.make_hint(refills=[1, 0])
        fields = hint_fields(hint)
        self.assertIsNotNone(fields)
        self.assertEqual(fields[8], [True, False])

    def test_hint_without_row_table_still_valid(self):
        fields = hint_fields(make_recv_hint(3, 4))
        self.assertIsNotNone(fields)
        self.assertEqual(fields[8], [])

    def test_row_table_validation_intact(self):
        # sum(rows) != num_tokens 依旧判死(防回归)
        self.assertIsNone(hint_fields(self.make_hint(num_tokens=6)))
        # rows 与 req_ids 不等长依旧判死
        self.assertIsNone(
            hint_fields(self.make_hint(rows=[5]))
        )


if __name__ == "__main__":
    unittest.main()
