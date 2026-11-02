"""Focused regression tests for Lesson 24 KL utilities."""

import unittest

import torch
import torch.nn.functional as F

from llmfp.rl.kl import categorical_kl_from_logits, kl_shaped_action_rewards


class KLPrimitivesTests(unittest.TestCase):
    def test_exact_categorical_kl_matches_two_response_example(self) -> None:
        p = torch.tensor([0.5491469396207161, 0.45085306037928385], dtype=torch.float64)
        q = torch.tensor([0.9, 0.1], dtype=torch.float64)
        result = categorical_kl_from_logits(p.log(), q.log())
        expected = (p * (p.log() - q.log())).sum()
        torch.testing.assert_close(result, expected)
        self.assertGreater(result.item(), 0)

    def test_identical_distributions_give_zero_and_keep_batch_dimensions(self) -> None:
        logits = torch.randn(2, 3, 5, generator=torch.Generator().manual_seed(24))
        result = categorical_kl_from_logits(logits, logits.clone())
        self.assertEqual(result.shape, (2, 3))
        torch.testing.assert_close(result, torch.zeros(2, 3), atol=2e-7, rtol=0)

    def test_categorical_kl_matches_analytic_gradient(self) -> None:
        current = torch.tensor([1.0, 0.5, -0.2], dtype=torch.float64, requires_grad=True)
        reference = torch.tensor([0.1, -0.4, 0.7], dtype=torch.float64)
        result = categorical_kl_from_logits(current, reference)
        gradient = torch.autograd.grad(result, current)[0]
        p = F.softmax(current.detach(), dim=-1)
        delta = F.log_softmax(current.detach(), dim=-1) - F.log_softmax(reference, dim=-1)
        expected_gradient = p * (delta - result.detach())
        torch.testing.assert_close(gradient, expected_gradient)

    def test_kl_validates_shape_dtype_and_finiteness(self) -> None:
        valid = torch.zeros(2, 3, 5)
        bad = [
            (valid, torch.zeros(2, 4, 5)),
            (valid, valid.double()),
            (valid.long(), valid.long()),
            (torch.empty(0, 3), torch.empty(0, 3)),
            (torch.tensor([float('nan'), 1.0]), torch.zeros(2)),
        ]
        for a, b in bad:
            with self.subTest(a=a.shape, b=b.shape), self.assertRaises(ValueError):
                categorical_kl_from_logits(a, b)

    def make_batch(self) -> tuple[torch.Tensor, ...]:
        # (B=2, T=4): log-probabilities of actual sampled actions.
        old = torch.tensor([[-0.1, -0.8, -0.2, 0.0], [-0.5, -1.5, 0.0, 0.0]])
        ref = torch.tensor([[-0.2, -0.3, -0.3, 0.0], [-0.4, -0.7, 0.0, 0.0]])
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        terminal = torch.tensor([2.0, 10.0])
        ended = torch.tensor([True, False])
        return old, ref, mask, terminal, ended

    def test_shaped_rewards_with_terminal_and_truncated_trajectories(self) -> None:
        old, ref, mask, terminal, ended = self.make_batch()
        actual = kl_shaped_action_rewards(old, ref, mask, terminal, ended, beta=0.5)
        expected = torch.tensor([[-0.05, 0.25, 1.95, 0.0], [0.05, 0.4, 0.0, 0.0]])
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (2, 4))
        torch.testing.assert_close(actual.sum(dim=1), torch.tensor([2.15, 0.45]))

    def test_shaped_rewards_allow_negative_log_ratios_and_no_penalty(self) -> None:
        old, ref, mask, terminal, ended = self.make_batch()
        actual = kl_shaped_action_rewards(old, ref, mask, terminal, ended, beta=0.5)
        self.assertGreater(actual[0, 1].item(), 0)
        no_beta = kl_shaped_action_rewards(old, ref, mask, terminal, ended, beta=0)
        torch.testing.assert_close(no_beta, torch.tensor([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))

    def test_equal_actor_and_reference_give_only_terminal_scores(self) -> None:
        old, _, mask, terminal, ended = self.make_batch()
        actual = kl_shaped_action_rewards(old, old.clone(), mask, terminal, ended, beta=0.8)
        torch.testing.assert_close(actual, torch.tensor([[0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0]]))

    def test_masked_padding_never_receives_reward(self) -> None:
        old, ref, mask, terminal, ended = self.make_batch()
        changed_old = old.clone()
        changed_ref = ref.clone()
        changed_old[~mask] = -10
        changed_ref[~mask] = -0.1
        baseline = kl_shaped_action_rewards(old, ref, mask, terminal, ended, beta=0.5)
        actual = kl_shaped_action_rewards(changed_old, changed_ref, mask, terminal, ended, beta=0.5)
        torch.testing.assert_close(actual, baseline)
        self.assertTrue((actual[~mask] == 0).all().item())

    def test_reward_shaping_detaches_from_autograd_and_does_not_mutate_inputs(self) -> None:
        old, ref, mask, terminal, ended = self.make_batch()
        old.requires_grad_()
        terminal.requires_grad_()
        original_old = old.detach().clone()
        actual = kl_shaped_action_rewards(old, ref, mask, terminal, ended, beta=0.2)
        self.assertFalse(actual.requires_grad)
        torch.testing.assert_close(old.detach(), original_old)
        self.assertIsNone(old.grad)
        self.assertIsNone(terminal.grad)

    def test_shaped_rewards_reject_invalid_beta(self) -> None:
        args = self.make_batch()
        for bad_beta in (-0.1, float('nan'), float('inf'), True):
            with self.subTest(beta=bad_beta), self.assertRaises(ValueError):
                kl_shaped_action_rewards(*args, beta=bad_beta)

    def test_shaped_rewards_reject_invalid_action_masks(self) -> None:
        old, ref, mask, terminal, ended = self.make_batch()
        bad_masks = [
            torch.zeros_like(mask),
            torch.tensor([[True, False, True, False], [True, True, False, False]]),
            mask.float(),
            mask[:, :-1],
        ]
        for bad in bad_masks:
            with self.subTest(mask=bad.shape), self.assertRaises(ValueError):
                kl_shaped_action_rewards(old, ref, bad, terminal, ended, beta=0.1)

    def test_shaped_rewards_reject_invalid_scores_and_dtypes(self) -> None:
        old, ref, mask, terminal, ended = self.make_batch()
        bad_cases = [
            (old, ref.float(), mask, terminal.double(), ended),
            (old, ref, mask, terminal[:1], ended),
            (old, ref, mask, terminal, ended[:1]),
            (old, ref, mask, terminal, ended.float()),
            (old, ref, mask, torch.tensor([float('nan'), 1.0]), ended),
        ]
        for values in bad_cases:
            with self.assertRaises(ValueError):
                kl_shaped_action_rewards(*values, beta=0.1)


if __name__ == '__main__':
    unittest.main()
