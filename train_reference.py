"""One-step HBP-LSTM training smoke test with synthetic data."""

import torch
import torch.nn.functional as F

from src.features import standardize_lma_sequence
from src.models import HBPLSTM


def main() -> None:
    torch.manual_seed(42)

    model = HBPLSTM(
        num_joints=17,
        coord_dim=3,
        high_level_dim=33,
        num_classes=7,
    )
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    pose = torch.randn(4, 60, 17, 3)
    lma = standardize_lma_sequence(torch.randn(4, 60, 33))
    labels = torch.randint(0, 7, (4,))

    output = model(pose, lma, sample=True)

    classification_loss = F.cross_entropy(output["logits"], labels)
    kl_weight = 1e-6
    loss = classification_loss + kl_weight * output["kl"]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        {
            "total_loss": float(loss.detach()),
            "classification_loss": float(classification_loss.detach()),
            "kl": float(output["kl"].detach()),
        }
    )


if __name__ == "__main__":
    main()
