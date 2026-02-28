import torch
import torch.nn as nn
import torch.nn.functional as F


class TruncatedLoss(nn.Module):
    """
    Generalised truncated cross-entropy loss for learning with noisy labels.

    Loss is clipped at a threshold k so that the gradient contribution from
    likely-corrupted samples (high loss) is zeroed out after the weight update.

    Reference: Huang et al., "O2U-Net: A Simple Noisy Label Detection Approach
    for Deep Neural Networks", ICCV 2019.
    """

    def __init__(self, q: float = 0.7, k: float = 0.5, trainset_size: int = 50000):
        super().__init__()
        self.q = q
        self.k = k
        self.weight = nn.Parameter(
            torch.ones(trainset_size, 1),
            requires_grad=False,
        )

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        indexes: torch.Tensor,
    ) -> torch.Tensor:
        eps = 1e-8
        p = F.softmax(logits, dim=1)
        p = torch.clamp(p, min=eps, max=1.0)

        Yg = torch.gather(p, 1, targets.unsqueeze(1))

        Lq = (1 - Yg.pow(self.q)) / self.q
        Lqk = (1 - self.k ** self.q) / self.q

        loss = (Lq - Lqk) * self.weight[indexes]
        return loss.mean()

    def update_weight(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        indexes: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            eps = 1e-8
            p = F.softmax(logits, dim=1)
            p = torch.clamp(p, min=eps, max=1.0)

            Yg = torch.gather(p, 1, targets.unsqueeze(1))

            Lq = (1 - Yg.pow(self.q)) / self.q
            Lqk = (1 - self.k ** self.q) / self.q

            condition = (Lq < Lqk).float()
            self.weight[indexes] = condition
