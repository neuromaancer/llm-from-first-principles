"""Regression tests for verified Lesson 23 preference-learning primitives."""

import math
import unittest

import torch
import torch.nn.functional as F

from llmfp.training.losses import (
    bradley_terry_loss,
    completion_sequence_log_probs,
    per_example_sft_nll,
)


class PreferencePrimitivesTests(unittest.TestCase):
    def test_completion_log_probs_match_notebook_masked_gather(self) -> None:
        # (B=2, T=4, V=5): reproducible causal logits.
        logits = torch.randn(
            2, 4, 5,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(23),
        )

        # (B=2, T=4): already aligned completion-only targets.
        labels = torch.tensor([
            [-100, 1, 2, -100],
            [-100, 2, -100, 4],
        ])
        mask = labels.ne(-100)
        safe_labels = labels.masked_fill(~mask, 0)

        # (B=2, T=4): notebook's log_softmax/gather reference.
        direct = F.log_softmax(logits, dim=-1).gather(
            -1,
            safe_labels.unsqueeze(-1),
        ).squeeze(-1)

        expected = direct.masked_fill(~mask, 0).sum(dim=1)
        actual = completion_sequence_log_probs(logits, labels)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (2,))

    def test_completion_log_probs_equal_negative_nll_times_length(self) -> None:
        logits = torch.randn(
            2, 4, 5,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(24),
        )
        labels = torch.tensor([
            [-100, 1, 2, -100],
            [-100, 2, -100, 4],
        ])
        actual = completion_sequence_log_probs(logits, labels)
        expected = (
            -per_example_sft_nll(logits, labels)
            * labels.ne(-100).sum(dim=1)
        )
        torch.testing.assert_close(actual, expected)

    def test_completion_log_probs_gradients_match_cross_entropy(self) -> None:
        # (B=2, T=4, V=5): differentiable logits.
        logits = torch.randn(
            2, 4, 5, dtype=torch.float64, requires_grad=True
        )
        labels = torch.tensor([
            [-100, 1, 2, -100],
            [-100, 2, -100, 4],
        ])
        score = completion_sequence_log_probs(logits, labels).sum()
        reference = -F.cross_entropy(
            logits.reshape(-1, 5),
            labels.reshape(-1),
            reduction="sum",
        )
        torch.testing.assert_close(score, reference)
        gradient = torch.autograd.grad(score, logits, retain_graph=True)[0]
        ref_gradient = torch.autograd.grad(reference, logits)[0]
        torch.testing.assert_close(gradient, ref_gradient)
        self.assertTrue(torch.all(gradient[labels.eq(-100)] == 0).item())
        self.assertGreater(gradient[labels.ne(-100)].norm().item(), 0)

    def test_ignored_logits_and_padding_do_not_change_scores(self) -> None:
        logits = torch.randn(2, 4, 5, dtype=torch.float64)
        labels = torch.tensor([
            [-100, 1, 2, -100],
            [-100, 2, -100, 4],
        ])
        baseline = completion_sequence_log_probs(logits, labels)

        modified_logits = logits.clone()
        modified_logits[labels.eq(-100)] = 70.0
        torch.testing.assert_close(
            completion_sequence_log_probs(modified_logits, labels),
            baseline,
        )

        padded_logits = torch.cat(
            (logits, torch.randn(2, 3, 5, dtype=torch.float64)),
            dim=1,
        )
        padded_labels = torch.cat(
            (labels, torch.full((2, 3), -100)),
            dim=1,
        )
        torch.testing.assert_close(
            completion_sequence_log_probs(padded_logits, padded_labels),
            baseline,
        )

    def test_custom_ignore_index(self) -> None:
        logits = torch.zeros(2, 2, 3)
        labels = torch.tensor([[-1, 0], [1, 2]])
        actual = completion_sequence_log_probs(
            logits, labels, ignore_index=-1
        )
        expected = torch.tensor([
            -math.log(3),
            -2 * math.log(3),
        ])
        torch.testing.assert_close(actual, expected)

    def test_completion_log_probs_reject_bad_contracts(self) -> None:
        logits = torch.zeros(2, 3, 4)
        bad_labels = [
            torch.full((2, 3), -100),
            torch.tensor([[1, -100, 2], [-100, -100, -100]]),
            torch.tensor([[4, 1, 2], [0, 1, 2]]),
            torch.tensor([[-1, 1, 2], [0, 1, 2]]),
            torch.zeros(2, 3),
            torch.zeros(2, 4, dtype=torch.long),
        ]
        for labels in bad_labels:
            with self.subTest(labels=labels):
                with self.assertRaises(ValueError):
                    completion_sequence_log_probs(logits, labels)

        with self.assertRaises(ValueError):
            completion_sequence_log_probs(
                logits.long(), torch.ones(2, 3, dtype=torch.long)
            )
        with self.assertRaises(ValueError):
            completion_sequence_log_probs(
                logits,
                torch.ones(2, 3, dtype=torch.long),
                ignore_index=0,
            )

    def test_bt_softplus_logsigmoid_and_reductions(self) -> None:
        # (B=3,): synthetic scalar reward scores from Lesson 23.04.
        chosen = torch.tensor([1.0, -1.0, 0.5], dtype=torch.float64)
        rejected = torch.tensor([0.0, 1.0, 0.5], dtype=torch.float64)
        expected = F.softplus(rejected - chosen)
        torch.testing.assert_close(
            bradley_terry_loss(chosen, rejected, reduction="none"),
            expected,
        )
        torch.testing.assert_close(
            bradley_terry_loss(chosen, rejected),
            expected.mean(),
        )
        torch.testing.assert_close(
            bradley_terry_loss(chosen, rejected, reduction="sum"),
            expected.sum(),
        )
        torch.testing.assert_close(
            bradley_terry_loss(chosen, rejected),
            -F.logsigmoid(chosen - rejected).mean(),
        )

    def test_bt_autograd_matches_analytical_gradients(self) -> None:
        chosen = torch.tensor(
            [1.0, -1.0, 0.5],
            dtype=torch.float64,
            requires_grad=True,
        )
        rejected = torch.tensor(
            [0.0, 1.0, 0.5],
            dtype=torch.float64,
            requires_grad=True,
        )
        loss = bradley_terry_loss(chosen, rejected)
        grad_chosen, grad_rejected = torch.autograd.grad(
            loss, (chosen, rejected)
        )
        probabilities = torch.sigmoid(
            chosen.detach() - rejected.detach()
        )
        torch.testing.assert_close(
            grad_chosen, (probabilities - 1) / 3
        )
        torch.testing.assert_close(
            grad_rejected, (1 - probabilities) / 3
        )
        torch.testing.assert_close(
            grad_chosen + grad_rejected,
            torch.zeros_like(grad_chosen),
        )

    def test_bt_common_shift_invariance_and_ties(self) -> None:
        chosen = torch.tensor([2.0, -2.0, 0.0], dtype=torch.float64)
        rejected = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
        offsets = torch.tensor([4.0, -5.0, 9.0])
        torch.testing.assert_close(
            bradley_terry_loss(chosen + offsets, rejected + offsets),
            bradley_terry_loss(chosen, rejected),
        )
        torch.testing.assert_close(
            bradley_terry_loss(
                torch.tensor([0.0]), torch.tensor([0.0])
            ),
            torch.tensor(math.log(2.0)),
        )

    def test_bt_stays_finite_with_extreme_reward_margins(self) -> None:
        chosen = torch.tensor([-1000.0, 1000.0])
        rejected = torch.tensor([1000.0, -1000.0])
        actual = bradley_terry_loss(
            chosen, rejected, reduction="none"
        )
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(
            actual,
            F.softplus(rejected - chosen),
        )

    def test_bt_rejects_invalid_reward_contracts(self) -> None:
        chosen = torch.tensor([1.0, 2.0])
        bad_pairs = [
            (chosen, torch.tensor([1.0])),
            (chosen.unsqueeze(-1), chosen.unsqueeze(-1)),
            (torch.empty(0), torch.empty(0)),
            (chosen, chosen.long()),
            (chosen, torch.tensor([1.0, float("nan")])),
        ]
        for positive, negative in bad_pairs:
            with self.subTest(positive=positive, negative=negative):
                with self.assertRaises(ValueError):
                    bradley_terry_loss(positive, negative)
        with self.assertRaises(ValueError):
            bradley_terry_loss(
                chosen, chosen, reduction="median"
            )

    def test_bt_updates_trainable_reward_head(self) -> None:
        # (B_candidates=4, C=2): deliberately non-canceling features.
        features = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ])
        head = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(head.weight)
        optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
        before = head.weight.detach().clone()

        # (B_candidates=4,): scalar reward predictions.
        scores = head(features).squeeze(-1)
        initial_loss = bradley_terry_loss(
            scores[0::2], scores[1::2]
        )
        optimizer.zero_grad(set_to_none=True)
        initial_loss.backward()
        optimizer.step()

        final_scores = head(features).squeeze(-1)
        final_loss = bradley_terry_loss(
            final_scores[0::2], final_scores[1::2]
        )
        self.assertLess(final_loss.item(), initial_loss.item())
        self.assertGreater((head.weight - before).norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
