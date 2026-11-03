"""A smoke test for the checkpoint-free source composition example."""

import math
import unittest

import torch

from examples.tiny_sft_dpo import run_demo


class TinyCompositionTests(unittest.TestCase):
    def test_example_composes_public_apis_without_checkpoints(self):
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(2)
            result = run_demo(sft_steps=4, dpo_steps=2)
        finally:
            torch.set_num_threads(threads)
        self.assertEqual(result['candidate_shape'], [6, 35])
        self.assertTrue(result['reference_unchanged'])
        self.assertAlmostEqual(result['dpo_loss']['before'], math.log(2), places=5)
        self.assertTrue(math.isfinite(result['dpo_loss']['after']))
        self.assertEqual(len(result['dpo_responses']), 3)


if __name__ == '__main__':
    unittest.main()
