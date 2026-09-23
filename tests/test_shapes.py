"""Interface tests for the public HBP-LSTM reference implementation."""

import unittest

import torch

from src.models import HBPLSTM
from src.robustness import add_gaussian_noise, fgsm_pose_attack


class HBPLSTMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.pose = torch.randn(2, 16, 17, 3)
        self.lma = torch.randn(2, 16, 33)
        self.labels = torch.tensor([0, 1])
        self.model = HBPLSTM(
            num_joints=17,
            coord_dim=3,
            high_level_dim=33,
            num_classes=7,
        )

    def test_forward_shapes(self):
        out = self.model(self.pose, self.lma, sample=False)
        self.assertEqual(tuple(out["logits"].shape), (2, 7))
        self.assertEqual(tuple(out["feature"].shape), (2, 128))
        self.assertEqual(out["kl"].ndim, 0)

    def test_mc_prediction_shapes(self):
        out = self.model.predict_mc(self.pose, self.lma, samples=3)
        self.assertEqual(tuple(out["mean_probability"].shape), (2, 7))
        self.assertEqual(tuple(out["predictive_entropy"].shape), (2,))

    def test_noise_shape(self):
        noisy = add_gaussian_noise(self.pose, sigma=0.01)
        self.assertEqual(tuple(noisy.shape), tuple(self.pose.shape))

    def test_fgsm_shape(self):
        adv = fgsm_pose_attack(
            self.model,
            self.pose,
            self.lma,
            self.labels,
            epsilon=0.001,
        )
        self.assertEqual(tuple(adv.shape), tuple(self.pose.shape))


if __name__ == "__main__":
    unittest.main()
