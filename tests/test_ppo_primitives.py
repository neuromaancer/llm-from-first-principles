"""PPO regression tests: math, gradients, rollout boundaries and token alignment."""

import math
import unittest

import torch

from llmfp.rl import (
    RolloutBatch,
    align_state_values_for_actions,
    categorical_entropy,
    generalized_advantage_estimation,
    masked_mean,
    ppo_policy_loss,
    ppo_value_loss,
    rescore_actions_from_logits,
    select_rollout_sequences,
)


def make_rollout_batch() -> RolloutBatch:
    """Construct one two-sequence, right-padded frozen toy rollout."""
    return RolloutBatch(
        token_ids=torch.tensor([[0, 1, 3, 4], [0, 2, 4, 0]]),
        action_token_positions=torch.tensor([[2, 3], [2, 0]]),
        action_mask=torch.tensor([[True, True], [True, False]]),
        old_log_probabilities=torch.zeros((2, 2)),
        rewards=torch.tensor([[0.0, 1.0], [0.5, 0.0]]),
        rollout_values=torch.zeros((2, 3)),
        bootstrap_mask=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        continuation_mask=torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        advantages=torch.tensor([[0.8, 1.0], [0.5, 0.0]]),
        value_targets=torch.tensor([[0.8, 1.0], [0.5, 0.0]]),
    )


class TestPPOPrimitives(unittest.TestCase):
    """Verify critical invariants of the Lesson 21A extraction."""

    def test_masked_mean_and_gradients(self) -> None:
        values = torch.tensor([2.0, 100.0, 6.0], requires_grad=True)
        mean = masked_mean(values, torch.tensor([True, False, True]))
        self.assertAlmostEqual(mean.item(), 4.0)
        mean.backward()
        torch.testing.assert_close(
            values.grad, torch.tensor([0.5, 0.0, 0.5])
        )

    def test_empty_mask_rejected(self) -> None:
        with self.assertRaises(ValueError):
            masked_mean(
                torch.tensor([1.0, 2.0]), torch.tensor([False, False])
            )

    def test_clipping_for_both_advantage_signs(self) -> None:
        log_ratios = torch.log(torch.tensor([1.5, 0.5]))
        loss, kl, fraction = ppo_policy_loss(
            log_ratios,
            torch.zeros(2),
            torch.tensor([2.0, -1.0]),
            torch.tensor([True, True]),
            0.2,
        )
        self.assertAlmostEqual(loss.item(), -0.8, places=6)
        self.assertGreater(kl.item(), 0.0)
        self.assertEqual(fraction.item(), 1.0)

    def test_actor_uses_only_current_gradients(self) -> None:
        current = torch.tensor([-0.5], requires_grad=True)
        old = torch.tensor([-0.5], requires_grad=True)
        advantage = torch.tensor([2.0], requires_grad=True)
        loss, _, _ = ppo_policy_loss(
            current, old, advantage, torch.tensor([True]), 0.2
        )
        loss.backward()
        self.assertAlmostEqual(current.grad.item(), -2.0)
        self.assertIsNone(old.grad)
        self.assertIsNone(advantage.grad)

    def test_value_loss_masks_padding_and_detaches_targets(self) -> None:
        current = torch.tensor([0.0, 0.0, 5.0], requires_grad=True)
        target = torch.tensor([1.0, -1.0, 100.0], requires_grad=True)
        loss = ppo_value_loss(
            current, target, torch.tensor([True, True, False])
        )
        self.assertAlmostEqual(loss.item(), 0.5)
        loss.backward()
        torch.testing.assert_close(
            current.grad, torch.tensor([-0.5, 0.5, 0.0])
        )
        self.assertIsNone(target.grad)

    def test_uniform_categorical_entropy(self) -> None:
        result = categorical_entropy(torch.zeros((2, 3)))
        torch.testing.assert_close(
            result, torch.full((2,), math.log(3.0))
        )

    def test_gaE_terminal_vs_truncation_and_detachment(self) -> None:
        rewards = torch.tensor([0.0, 0.0, 1.0])
        values = torch.tensor(
            [0.2, 0.3, 0.4, 10.0], requires_grad=True
        )
        continuation = torch.tensor([1.0, 1.0, 0.0])
        terminal, targets = generalized_advantage_estimation(
            rewards,
            values,
            torch.tensor([1.0, 1.0, 0.0]),
            continuation,
            0.9,
            0.95,
        )
        truncated, _ = generalized_advantage_estimation(
            rewards,
            values,
            torch.tensor([1.0, 1.0, 1.0]),
            continuation,
            0.9,
            0.95,
        )
        self.assertAlmostEqual(terminal[-1].item(), 0.6, places=5)
        self.assertAlmostEqual(truncated[-1].item(), 9.6, places=5)
        self.assertAlmostEqual(targets[-1].item(), 1.0, places=5)
        self.assertFalse(terminal.requires_grad)

    def test_invalid_gae_boundary_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generalized_advantage_estimation(
                torch.tensor([1.0]),
                torch.tensor([0.0, 0.0]),
                torch.tensor([0.0]),
                torch.tensor([1.0]),
                0.9,
                0.95,
            )

    def test_causal_rescoring_index_and_gradient(self) -> None:
        logits = torch.zeros((1, 4, 5), requires_grad=True)
        result = rescore_actions_from_logits(
            logits,
            torch.tensor([[0, 1, 3, 4]]),
            torch.tensor([[2, 3]]),
            torch.tensor([[True, True]]),
        )
        torch.testing.assert_close(
            result, torch.full((1, 2), -math.log(5.0))
        )
        result.sum().backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0.0)
        self.assertEqual(logits.grad[0, 3].abs().sum().item(), 0.0)
        self.assertGreater(logits.grad[0, 1].abs().sum().item(), 0.0)

    def test_causal_rescoring_padding(self) -> None:
        logits = torch.zeros((1, 4, 5), requires_grad=True)
        result = rescore_actions_from_logits(
            logits,
            torch.tensor([[0, 1, 3, 0]]),
            torch.tensor([[2, 0]]),
            torch.tensor([[True, False]]),
        )
        self.assertEqual(result[0, 1].item(), 0.0)
        result.sum().backward()
        self.assertEqual(logits.grad[0, 0].abs().sum().item(), 0.0)

    def test_invalid_action_positions_and_ids_rejected(self) -> None:
        logits = torch.zeros((1, 4, 5))
        ids = torch.tensor([[0, 1, 3, 4]])
        mask = torch.tensor([[True]])
        for position in (0, 4):
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    rescore_actions_from_logits(
                        logits, ids, torch.tensor([[position]]), mask
                    )
        with self.assertRaises(ValueError):
            rescore_actions_from_logits(
                logits,
                torch.tensor([[0, 1, 3, 7]]),
                torch.tensor([[3]]),
                mask,
            )

    def test_critic_values_with_right_padding(self) -> None:
        values = torch.arange(5, dtype=torch.float32).expand(2, -1)
        result = align_state_values_for_actions(
            values,
            torch.tensor([[3, 4], [3, 0]]),
            torch.tensor([[True, True], [True, False]]),
        )
        torch.testing.assert_close(
            result, torch.tensor([[2., 3., 4.], [2., 3., 0.]])
        )

    def test_alignment_rejects_nonconsecutive_and_holey_actions(self) -> None:
        values = torch.zeros((1, 6))
        for positions, mask in (
            ([[2, 4]], [[True, True]]),
            ([[2, 0, 4]], [[True, False, True]]),
        ):
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError):
                    align_state_values_for_actions(
                        values, torch.tensor(positions), torch.tensor(mask)
                    )

    def test_minibatch_selection_preserves_all_fields(self) -> None:
        batch = make_rollout_batch()
        subbatch = select_rollout_sequences(batch, torch.tensor([1]))
        self.assertEqual(subbatch.token_ids.shape, (1, 4))
        torch.testing.assert_close(subbatch.rewards, batch.rewards[[1]])
        self.assertEqual(subbatch.action_mask.sum().item(), 1)

    def test_negative_minibatch_indices_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_rollout_sequences(
                make_rollout_batch(), torch.tensor([-1])
            )

    def test_rollout_rejects_non_detached_collection_data(self) -> None:
        batch = make_rollout_batch()
        with self.assertRaises(ValueError):
            RolloutBatch(
                token_ids=batch.token_ids,
                action_token_positions=batch.action_token_positions,
                action_mask=batch.action_mask,
                old_log_probabilities=torch.zeros(
                    (2, 2), requires_grad=True
                ),
                rewards=batch.rewards,
                rollout_values=batch.rollout_values,
                bootstrap_mask=batch.bootstrap_mask,
                continuation_mask=batch.continuation_mask,
                advantages=batch.advantages,
                value_targets=batch.value_targets,
            )


if __name__ == "__main__":
    unittest.main()
