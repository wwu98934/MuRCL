import math
import torch
import torch.nn.functional as F
from torch import nn


class ABMIL(nn.Module):
    def __init__(self, dim_in, L=512, D=128, K=1, dim_out=2, dropout=0.):
        super(ABMIL, self).__init__()
        self.L, self.D, self.K = L, D, K

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, self.L),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.L, self.L),
            nn.ReLU(),
        )
        self.fc = nn.Linear(self.L, dim_out)

    def bag_forward(self, bag):
        H = self.encoder(bag)  # NxL

        A = self.attention(H)  # Nx1
        A = torch.transpose(A, 1, 0)  # 1xN
        A = F.softmax(A, dim=1)  # softmax over N
        A = A / math.sqrt(A.shape[-1])
        M = torch.mm(A, H)  # 1xL

        output = self.decoder(M)
        return output

    def batch_forward(self, batch):
        outputs = []
        for bag in batch:
            outputs.append(self.bag_forward(bag))
        return torch.cat(outputs, 0)

    def forward(self, x):  # B x N x dim_in, a bag
        if isinstance(x, list):
            outputs = self.batch_forward(x)
        elif isinstance(x, torch.Tensor):
            if x.shape[0] == 1:
                outputs = self.bag_forward(x.squeeze(0))
            else:
                outputs = self.batch_forward(x)
        else:
            raise TypeError
        return outputs, outputs.detach()
