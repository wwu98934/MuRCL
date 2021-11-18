import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def neg_partial_log(prediction, T, E):
    """
    calculate cox loss, Pytorch implementation by Huang, https://github.com/huangzhii/SALMON
    :param prediction: variables
    :param T: Time
    :param E: Status
    :return: neg log of the likelihood
    """

    current_batch_len = len(prediction)
    # print(current_batch_len)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train).cuda()
    train_y_status = torch.tensor(E, dtype=torch.float).cuda()

    theta = prediction.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_y_status)

    return loss_nn


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=torch.bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size
        print(f"N = {N}")

        z = torch.cat((z_i, z_j), dim=0)
        print(f"z:{z.shape}")

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        print(f"z.unsqueeze(1):{z.unsqueeze(1).shape}")
        print(f"z.unsqueeze(0):{z.unsqueeze(0).shape}")
        print(f"sim:{sim.shape}")

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        print(f"sim_i_j:{sim_i_j.shape}")
        print(f"sim_j_i:{sim_j_i.shape}")

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        print(f"positive_samples:{positive_samples.shape}")
        print(f"negative_samples:{negative_samples.shape}")

        labels = torch.zeros(N).to(positive_samples.device).long()
        print(f"labels:{labels.shape}")
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        print(f"logits:{logits.shape}")
        loss = self.criterion(logits, labels)
        loss /= N
        return loss


# Test -----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    nt_loss = NT_Xent(128, 1.0)
    x_i = torch.randn(size=(128, 128))
    x_j = torch.randn(size=(128, 128))
    loss = nt_loss(x_i, x_j)
    print(loss)
    ...
