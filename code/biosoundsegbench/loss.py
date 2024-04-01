import torch
import torch.nn.functional as F


class CrossEntropyWithTruncatedMSE(torch.nn.Module):
    def __init__(self, tmse_weight=0.15):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss(reduction='none')
        self.tmse_weight = tmse_weight

    def forward(self, input, target):
        ce = self.ce(input, target)
        mse = self.tmse_weight * torch.mean(
            torch.clamp(
                self.mse(
                    F.log_softmax(input[:, :, 1:], dim=1), F.log_softmax(input.detach()[:, :, :-1], dim=1)
                ), min=0, max=16)
            )
        return ce + mse


class CrossEntropyWithGaussianSimilarityTruncatedMSE(torch.nn.Module):
    def __init__(self, gstmse_weight=0.15, threshold=4, sigma=1.0):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss(reduction='none')
        self.gstmse_weight = gstmse_weight
        self.threshold = threshold
        self.sigma = sigma

    def forward(self, input, target, x):
        ce = self.ce(input, target)

        # calculate temporal mse
        mse = self.mse(
            F.log_softmax(input[:, :, 1:], dim=1), F.log_softmax(input[:, :, :-1], dim=1)
        )
        mse = torch.clamp(mse, min=0, max=self.threshold ** 2)

        # calculate gaussian similarity
        x = torch.squeeze(x, dim=1)
        diff = x[:, :, 1:] - x[:, :, :-1]
        similarity = torch.exp(-torch.norm(diff, dim=1) / (2 * self.sigma ** 2))
        # gaussian similarity weighting
        similarity = torch.unsqueeze(similarity, dim=1)  # for broadcasting
        mse = self.gstmse_weight * torch.mean(similarity * mse)

        return ce + mse
