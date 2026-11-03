"""Numerical boundary regressions; no training benchmark."""

import unittest

import torch

from llmfp.rl import group_relative_advantages, grpo_clipped_policy_loss


class GroupRelativeBoundaryTests(unittest.TestCase):
    def test_constant_decimal_rewards_produce_exact_zero(self):
        for dtype in (torch.float16, torch.float32, torch.float64):
            for normalize in (False, True):
                with self.subTest(dtype=dtype, normalize=normalize):
                    rewards = torch.full((2, 16), 0.1, dtype=dtype)
                    result = group_relative_advantages(rewards, normalize=normalize)
                    self.assertTrue(torch.isfinite(result).all().item())
                    self.assertTrue(result.eq(0).all().item())

    def test_mixed_and_constant_prompts_remain_independent(self):
        rewards = torch.tensor([[1., 0., 0.], [0.1, 0.1, 0.1]])
        result = group_relative_advantages(rewards)
        expected = torch.tensor([[2**0.5, -2**0.5 / 2, -2**0.5 / 2], [0., 0., 0.]])
        torch.testing.assert_close(result, expected)

    def test_padding_cannot_create_nonfinite_gradients(self):
        current = torch.tensor([[[-1., 0.], [-1., 0.]]], requires_grad=True)
        old = torch.tensor([[[-1., -1000.], [-1., -1000.]]])
        mask = torch.tensor([[[True, False], [True, False]]])
        loss = grpo_clipped_policy_loss(current, old, torch.tensor([[1., -1.]]), mask)
        loss.backward()
        self.assertTrue(torch.isfinite(current.grad).all().item())
        self.assertTrue(current.grad[~mask].eq(0).all().item())
        torch.testing.assert_close(current.grad[mask], torch.tensor([-0.5, 0.5]))


if __name__ == '__main__':
    unittest.main()
