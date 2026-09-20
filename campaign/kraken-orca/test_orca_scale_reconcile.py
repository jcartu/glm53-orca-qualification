"""CPU-only self-check for the NVFP4 gate/up reconciliation helper."""

from __future__ import annotations

import unittest

import torch

from orca_scale_reconcile import ScaleReconciliationError, reconcile_gate_up_scales


def effective(block: torch.Tensor, divisor: float) -> torch.Tensor:
    return block.float() / divisor


class ReconcileTest(unittest.TestCase):
    def test_matching_pairs_are_untouched(self):
        scales = torch.full((2, 4, 2), 8.0, dtype=torch.float8_e4m3fn)
        globals_ = torch.tensor([[1024.0, 1024.0], [2048.0, 2048.0]])
        out_scales, out_globals, stats = reconcile_gate_up_scales(scales, globals_)
        self.assertEqual(stats["mismatched_experts"], 0.0)
        self.assertTrue(torch.equal(out_scales.float(), scales.float()))
        self.assertTrue(torch.equal(out_globals, globals_))

    def test_mismatch_preserves_effective_dequant_within_e4m3_rounding(self):
        gate_block = torch.tensor([[[44.0, 88.0], [44.0, 88.0]]], dtype=torch.float32)
        up_block = torch.tensor([[[52.0, 96.0], [52.0, 96.0]]], dtype=torch.float32)
        scales = (
            torch.cat((gate_block, up_block), dim=1)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        globals_ = torch.tensor([[24576.0, 26496.0]])
        out_scales, out_globals, stats = reconcile_gate_up_scales(scales, globals_)
        self.assertEqual(stats["mismatched_experts"], 1.0)
        self.assertTrue(torch.equal(out_globals[:, 0], out_globals[:, 1]))
        unified = float(out_globals[0, 0])
        self.assertEqual(unified, 24576.0)
        out_gate = out_scales[:, 0:2, :].float()
        out_up = out_scales[:, 2:4, :].float()
        gate_error = (
            (
                effective(out_gate, unified)
                - effective(gate_block.to(torch.float8_e4m3fn).float(), 24576.0)
            )
            .abs()
            .max()
        )
        up_relative = (
            (
                effective(out_up, unified)
                - effective(up_block.to(torch.float8_e4m3fn).float(), 26496.0)
            ).abs()
            / effective(up_block.to(torch.float8_e4m3fn).float(), 26496.0).abs()
        ).max()
        self.assertEqual(gate_error, 0.0)
        self.assertLessEqual(float(up_relative), 0.0625)

    def test_underflow_to_zero_is_refused(self):
        tiny = torch.tensor([[[2.0**-9, 8.0], [2.0**-9, 8.0]]], dtype=torch.float32)
        scales = torch.cat((tiny, tiny), dim=1).to(torch.float8_e4m3fn).contiguous()
        globals_ = torch.tensor([[1024.0, 1024.0 * 32.0]])
        with self.assertRaises(ScaleReconciliationError):
            reconcile_gate_up_scales(scales, globals_)

    def test_nonpositive_globals_are_refused(self):
        scales = torch.full((1, 2, 2), 8.0, dtype=torch.float8_e4m3fn)
        globals_ = torch.tensor([[1024.0, -4.0]])
        with self.assertRaises(ScaleReconciliationError):
            reconcile_gate_up_scales(scales, globals_)

    def test_shrink_factor_is_bounded_by_divisor_ratio(self):
        scales = torch.full((1, 4, 2), 16.0, dtype=torch.float8_e4m3fn)
        globals_ = torch.tensor([[4096.0, 1024.0]])
        _, _, stats = reconcile_gate_up_scales(scales, globals_)
        self.assertAlmostEqual(stats["max_shrink_factor"], 4.0, places=6)
        self.assertLessEqual(stats["max_block_scale_after"], 448.0)


if __name__ == "__main__":
    unittest.main()
