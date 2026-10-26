"""Focused regression checks for the primitives learned in Lesson 22."""

import math
import unittest

import torch
import torch.nn.functional as F

from llmfp.data.supervision import pad_supervised_batch, shift_supervised_tokens
from llmfp.training.losses import per_example_sft_nll


class SupervisedPrimitivesTests(unittest.TestCase):
    def make_examples(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Return the two symbolic examples used in Lesson 22.04."""
        # (L_i,): full conversations and selected full-sequence target tokens.
        short_ids = torch.tensor([0, 1, 2, 3, 4, 5])
        short_mask = torch.tensor([False, False, False, False, True, True])
        long_ids = torch.tensor([0, 1, 2, 6, 3, 4, 7, 5])
        long_mask = torch.tensor([False] * 5 + [True] * 3)
        return [
            shift_supervised_tokens(short_ids, short_mask),
            shift_supervised_tokens(long_ids, long_mask),
        ]

    def test_shift_matches_notebook_and_keeps_inputs_independent(self) -> None:
        ids = torch.tensor([0, 1, 2, 3, 4, 5])
        original = ids.clone()
        mask = torch.tensor([False, False, False, False, True, True])
        inputs, labels = shift_supervised_tokens(ids, mask)
        self.assertEqual(inputs.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(labels.tolist(), [-100, -100, -100, 4, 5])
        inputs[0] = 7
        labels[-1] = 7
        torch.testing.assert_close(ids, original)

    def test_shift_does_not_assume_role_ids_or_contiguous_supervision(self) -> None:
        ids = torch.tensor([8, 9, 10, 11, 12])
        mask = torch.tensor([True, False, True, False, True])
        inputs, labels = shift_supervised_tokens(ids, mask, ignore_index=-1)
        self.assertEqual(inputs.tolist(), [8, 9, 10, 11])
        self.assertEqual(labels.tolist(), [-1, 10, -1, 12])

    def test_shift_rejects_bad_contracts(self) -> None:
        ids = torch.tensor([0, 1, 2])
        cases = [
            (ids, torch.tensor([True, False, False])),
            (ids, torch.tensor([False, False, False])),
            (ids, torch.ones(3)),
            (ids, torch.ones(2, dtype=torch.bool)),
            (ids.float(), torch.ones(3, dtype=torch.bool)),
            (torch.tensor([0, -100, 2]), torch.ones(3, dtype=torch.bool)),
            (torch.tensor([0]), torch.tensor([True])),
        ]
        for bad_ids, bad_mask in cases:
            with self.subTest(ids=bad_ids, mask=bad_mask), self.assertRaises(ValueError):
                shift_supervised_tokens(bad_ids, bad_mask)
        with self.assertRaises(ValueError):
            shift_supervised_tokens(ids, torch.ones(3, dtype=torch.bool), ignore_index=0)

    def test_padding_matches_notebook_without_double_shift(self) -> None:
        examples = self.make_examples()
        inputs, labels = pad_supervised_batch(examples, pad_token_id=0)
        self.assertEqual(inputs.shape, (2, 7))
        self.assertEqual(inputs[0].tolist(), [0, 1, 2, 3, 4, 0, 0])
        self.assertEqual(labels[0].tolist(), [-100, -100, -100, 4, 5, -100, -100])
        self.assertEqual(labels[1].tolist(), [-100, -100, -100, -100, 4, 7, 5])
        self.assertEqual(labels[labels.ne(-100)].tolist(), [4, 5, 4, 7, 5])
        self.assertEqual(examples[0][0].numel(), 5)

    def test_padding_preserves_real_tokens_with_placeholder_id(self) -> None:
        examples = [(torch.tensor([0, 0]), torch.tensor([0, 1])),
                    (torch.tensor([2]), torch.tensor([0]))]
        inputs, labels = pad_supervised_batch(examples, pad_token_id=0)
        self.assertEqual(inputs.tolist(), [[0, 0], [2, 0]])
        self.assertEqual(labels.tolist(), [[0, 1], [0, -100]])

    def test_padding_rejects_empty_or_malformed_examples(self) -> None:
        valid = self.make_examples()
        cases = [[], [(torch.tensor([0]), torch.tensor([-100]))],
                 [(torch.tensor([0, 1]), torch.tensor([1]))],
                 [(torch.tensor([-100]), torch.tensor([1]))],
                 [(torch.tensor([0]), torch.tensor([-2]))],
                 [(torch.tensor([0.0]), torch.tensor([1]))]]
        for examples in cases:
            with self.subTest(examples=examples), self.assertRaises(ValueError):
                pad_supervised_batch(examples, pad_token_id=0)
        with self.assertRaises(ValueError):
            pad_supervised_batch(valid, pad_token_id=-100)
        with self.assertRaises(ValueError):
            pad_supervised_batch(valid, pad_token_id=0, ignore_index=1)

    def test_per_example_matches_notebook_formula(self) -> None:
        _, labels = pad_supervised_batch(self.make_examples(), pad_token_id=0)
        # (B=2, T=7, V=8): reproducible, nonuniform logits.
        logits = torch.randn(2, 7, 8, generator=torch.Generator().manual_seed(22))
        actual = per_example_sft_nll(logits, labels)
        expected = torch.stack([
            F.cross_entropy(row_logits[row_labels.ne(-100)], row_labels[row_labels.ne(-100)])
            for row_logits, row_labels in zip(logits, labels)
        ])
        torch.testing.assert_close(actual, expected)

    def test_boolean_and_ignore_losses_and_gradients_match(self) -> None:
        _, labels = pad_supervised_batch(self.make_examples(), pad_token_id=0)
        mask = labels.ne(-100)
        logits = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
        counts = mask.sum(dim=1)
        loss = (per_example_sft_nll(logits, labels) * counts).sum() / counts.sum()
        reference = F.cross_entropy(logits[mask], labels[mask])
        direct = F.cross_entropy(logits.reshape(-1, 8), labels.reshape(-1), ignore_index=-100)
        torch.testing.assert_close(loss, reference)
        torch.testing.assert_close(loss, direct)
        grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        ref_grad = torch.autograd.grad(reference, logits)[0]
        torch.testing.assert_close(grad, ref_grad)
        self.assertTrue(torch.all(grad[~mask] == 0).item())
        self.assertGreater(grad[mask].norm().item(), 0)

    def test_token_and_example_normalizations_are_different(self) -> None:
        logits = torch.tensor([[[2.0, 0.0], [0.0, 0.0]],
                               [[0.0, 0.0], [0.0, 0.0]]])
        labels = torch.tensor([[0, -100], [0, 1]])
        losses = per_example_sft_nll(logits, labels)
        counts = labels.ne(-100).sum(dim=1)
        token_mean = (losses * counts).sum() / counts.sum()
        direct = F.cross_entropy(logits.reshape(-1, 2), labels.reshape(-1))
        torch.testing.assert_close(token_mean, direct)
        self.assertFalse(torch.isclose(losses.mean(), token_mean).item())

    def test_extra_ignored_positions_leave_loss_unchanged(self) -> None:
        logits = torch.zeros(2, 2, 3)
        labels = torch.tensor([[0, -100], [1, 2]])
        padded_logits = torch.cat([logits, torch.randn(2, 4, 3)], dim=1)
        padded_labels = torch.cat([labels, torch.full((2, 4), -100)], dim=1)
        torch.testing.assert_close(per_example_sft_nll(logits, labels),
                                   per_example_sft_nll(padded_logits, padded_labels))
        torch.testing.assert_close(per_example_sft_nll(logits, labels),
                                   torch.full((2,), math.log(3)))

    def test_loss_rejects_empty_rows_and_invalid_targets(self) -> None:
        logits = torch.zeros(2, 2, 3)
        cases = [torch.full((2, 2), -100), torch.tensor([[0, 1], [-100, -100]]),
                 torch.tensor([[0, 3], [1, 2]]), torch.tensor([[0, -1], [1, 2]]),
                 torch.zeros(2, 2), torch.zeros(2, 3, dtype=torch.long)]
        for labels in cases:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                per_example_sft_nll(logits, labels)
        with self.assertRaises(ValueError):
            per_example_sft_nll(torch.empty(0, 2, 3), torch.empty(0, 2, dtype=torch.long))
        with self.assertRaises(ValueError):
            per_example_sft_nll(logits.long(), torch.zeros(2, 2, dtype=torch.long))

    def test_differentiable_loss_updates_model_and_supports_custom_sentinel(self) -> None:
        model = torch.nn.Linear(3, 4, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.ones(1, 3, 3)
        labels = torch.tensor([[-1, 1, 2]])
        before = model.weight.detach().clone()
        loss = per_example_sft_nll(model(inputs), labels, ignore_index=-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        self.assertGreater((model.weight.detach() - before).norm().item(), 0)


if __name__ == "__main__":
    unittest.main()
