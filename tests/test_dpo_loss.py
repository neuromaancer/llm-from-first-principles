"""Minimal regression tests for the Lesson 25 DPO objective."""

import math
import unittest

import torch
import torch.nn.functional as F

from llmfp.training.losses import dpo_loss


class DPOLossTests(unittest.TestCase):
    def test_initial_log2_and_analytic_policy_gradients(self) -> None:
        chosen = torch.tensor([-1.0, -2.0], dtype=torch.float64, requires_grad=True)
        rejected = torch.tensor([-2.0, -1.0], dtype=torch.float64, requires_grad=True)
        loss = dpo_loss(chosen, rejected, chosen.detach(), rejected.detach(), beta=0.5)
        torch.testing.assert_close(loss, torch.tensor(math.log(2.0), dtype=torch.float64))
        grad_chosen, grad_rejected = torch.autograd.grad(loss, (chosen, rejected))
        torch.testing.assert_close(grad_chosen, torch.full((2,), -0.125, dtype=torch.float64))
        torch.testing.assert_close(grad_rejected, torch.full((2,), 0.125, dtype=torch.float64))

    def test_relative_margin_and_reductions(self) -> None:
        p_chosen = torch.tensor([0.20, 0.30], dtype=torch.float64).log()
        p_rejected = torch.tensor([0.80, 0.70], dtype=torch.float64).log()
        ref_chosen = torch.tensor([0.05, 0.50], dtype=torch.float64).log()
        ref_rejected = torch.tensor([0.95, 0.50], dtype=torch.float64).log()
        expected_logits = 0.5 * ((p_chosen - p_rejected) - (ref_chosen - ref_rejected))
        expected = -F.logsigmoid(expected_logits)
        args = (p_chosen, p_rejected, ref_chosen, ref_rejected)
        torch.testing.assert_close(dpo_loss(*args, beta=0.5, reduction="none"), expected)
        torch.testing.assert_close(dpo_loss(*args, beta=0.5), expected.mean())
        torch.testing.assert_close(dpo_loss(*args, beta=0.5, reduction="sum"), expected.sum())
        self.assertAlmostEqual(expected[0].item(), 0.3776357506966529, places=12)

    def test_reference_scores_do_not_receive_gradients(self) -> None:
        p_chosen = torch.tensor([-2.0], requires_grad=True)
        p_rejected = torch.tensor([-3.0], requires_grad=True)
        ref_chosen = torch.tensor([-1.0], requires_grad=True)
        ref_rejected = torch.tensor([-2.0], requires_grad=True)
        loss = dpo_loss(p_chosen, p_rejected, ref_chosen, ref_rejected)
        loss.backward()
        self.assertIsNotNone(p_chosen.grad)
        self.assertIsNotNone(p_rejected.grad)
        self.assertIsNone(ref_chosen.grad)
        self.assertIsNone(ref_rejected.grad)

    def test_extreme_margins_remain_finite(self) -> None:
        chosen = torch.tensor([-1000.0, -0.1])
        rejected = torch.tensor([-0.1, -1000.0])
        reference = torch.tensor([-1.0, -1.0])
        result = dpo_loss(chosen, rejected, reference, reference, beta=0.5, reduction="none")
        self.assertTrue(torch.isfinite(result).all().item())
        self.assertGreater(result[0].item(), 400)
        self.assertLess(result[1].item(), 1e-6)

    def test_rejects_invalid_inputs(self) -> None:
        values = torch.tensor([-1.0, -2.0])
        for beta in (0, -0.1, float('nan'), float('inf'), True):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                dpo_loss(values, values, values, values, beta=beta)
        bad = (
            torch.tensor([-1.0]),
            values.double(),
            torch.tensor([-1.0, float('nan')]),
            values.long(),
        )
        for tensor in bad:
            with self.subTest(tensor=tensor), self.assertRaises(ValueError):
                dpo_loss(values, tensor, values, values)
        with self.assertRaises(ValueError):
            dpo_loss(values, values, values, values, reduction="median")


if __name__ == "__main__":
    unittest.main()
