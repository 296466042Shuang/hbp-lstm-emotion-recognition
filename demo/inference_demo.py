"""Minimal HBP-LSTM inference demo using synthetic inputs."""

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import standardize_lma_sequence
from src.models import HBPLSTM


def main() -> None:
    torch.manual_seed(7)

    model = HBPLSTM(
        num_joints=17,
        coord_dim=3,
        high_level_dim=33,
        num_classes=7,
    )
    model.eval()

    pose = torch.randn(2, 60, 17, 3)
    lma = standardize_lma_sequence(torch.randn(2, 60, 33))

    with torch.no_grad():
        deterministic = model(pose, lma, sample=False)
        uncertainty = model.predict_mc(pose, lma, samples=10)

    print("pose:", tuple(pose.shape))
    print("LMA:", tuple(lma.shape))
    print("logits:", tuple(deterministic["logits"].shape))
    print("predictive entropy:", uncertainty["predictive_entropy"])


if __name__ == "__main__":
    main()
