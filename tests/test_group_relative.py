"""Focused tests for Lesson 26 group-relative RL primitives."""

import unittest

import torch

from llmfp.rl.group_relative import group_relative_advantages, grpo_clipped_policy_loss


class GroupRelativeTests(unittest.TestCase):
    def test_group_advantages_match_lesson_and_zero_variance(self):
        rewards = torch.tensor([[1., 0., 0.], [0., 0., 0.]])
        expected = torch.tensor([[2**0.5, -2**0.5 / 2, -2**0.5 / 2], [0., 0., 0.]])
        torch.testing.assert_close(group_relative_advantages(rewards), expected)
        torch.testing.assert_close(group_relative_advantages(rewards, normalize=False), rewards - rewards.mean(dim=1, keepdim=True))

    def test_advantages_are_frozen_and_prompt_separated(self):
        rewards = torch.tensor([[1., 0.], [10., 10.]], requires_grad=True)
        centered = group_relative_advantages(rewards, normalize=False)
        torch.testing.assert_close(centered, torch.tensor([[0.5, -0.5], [0., 0.]]))
        self.assertFalse(centered.requires_grad)

    def test_original_grpo_per_response_length_weighting(self):
        old = torch.full((1, 2, 6), -1.)
        current = old.detach().clone().requires_grad_(True)
        mask = torch.tensor([[[True, True, False, False, False, False], [True]*6]])
        advantage = torch.tensor([[1., -1.]])
        loss = grpo_clipped_policy_loss(current, old, advantage, mask)
        grad = torch.autograd.grad(loss, current)[0]
        self.assertAlmostEqual(grad[0, 0, 0].item(), -0.25)
        self.assertAlmostEqual(grad[0, 1, 0].item(), 1 / 12)
        self.assertTrue((grad[~mask] == 0).all().item())

    def test_clip_higher_preserves_the_positive_gradient(self):
        old = torch.full((1, 2, 1), -1.)
        current = old.clone()
        current[0, 0, 0] += torch.tensor(1.25).log()
        current.requires_grad_(True)
        adv = torch.tensor([[1., -1.]])
        mask = torch.ones((1, 2, 1), dtype=torch.bool)
        sym = grpo_clipped_policy_loss(current, old, adv, mask)
        asym = grpo_clipped_policy_loss(current, old, adv, mask, clip_epsilon_high=0.28)
        grad_sym = torch.autograd.grad(sym, current, retain_graph=True)[0]
        grad_asym = torch.autograd.grad(asym, current)[0]
        self.assertAlmostEqual(grad_sym[0, 0, 0].item(), 0.)
        self.assertAlmostEqual(grad_asym[0, 0, 0].item(), -0.625)

    def test_policy_gradients_not_old_or_advantage(self):
        old = torch.full((1, 2, 2), -1., requires_grad=True)
        current = old.detach().clone().requires_grad_(True)
        adv = torch.tensor([[1., -1.]], requires_grad=True)
        mask = torch.ones_like(current, dtype=torch.bool)
        loss = grpo_clipped_policy_loss(current, old, adv, mask)
        loss.backward()
        self.assertIsNotNone(current.grad)
        self.assertGreater(current.grad.norm().item(), 0)
        self.assertIsNone(old.grad)
        self.assertIsNone(adv.grad)

    def test_invalid_shapes_reward_types_and_masks(self):
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.ones(3))
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.ones(2, 1))
        with self.assertRaises(ValueError):
            group_relative_advantages(torch.tensor([[float('nan'), 1.]]))
        current = torch.full((1, 2, 2), -1.)
        adv = torch.tensor([[1., -1.]])
        mask = torch.tensor([[[True, False], [True, True]]])
        with self.assertRaises(ValueError):
            grpo_clipped_policy_loss(current, current, adv, mask.float())
        with self.assertRaises(ValueError):
            grpo_clipped_policy_loss(current, current, adv, torch.tensor([[[False, True], [True, True]]]))
        with self.assertRaises(ValueError):
            grpo_clipped_policy_loss(current, current, adv, mask, clip_epsilon_high=-0.1)


if __name__ == '__main__':
    unittest.main()
