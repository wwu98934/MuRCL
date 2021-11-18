import torch
import torch.nn.functional as F
from torch import nn


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def bag_forward(self, feats):
        assert len(feats.shape) == 3 and feats.shape[0] == 1, f"feats.shape: {feats.shape}"
        feats = feats.squeeze(0)
        feats = feats.cuda()
        x = self.fc(feats)
        return feats, x

    def batch_forward(self, feats):
        feats_list, x_list = [], []
        for f in feats:
            feats_forward, x_forward = self.bag_forward(f)
            feats_list.append(feats_forward)
            x_list.append(x_forward)
        return feats_list, x_list

    def forward(self, feats):
        if isinstance(feats, torch.Tensor) and len(feats.shape) == 3 and feats.shape[0] == 1:
            feats, x = self.bag_forward(feats)
        elif isinstance(feats, torch.Tensor) and len(feats.shape) == 3 and feats.shape[0] > 1:
            feats = [feats[i, :, :].unsqueeze(0) for i in range(feats.shape[0])]
            feats, x = self.batch_forward(feats)
        elif isinstance(feats, list):
            feats, x = self.batch_forward(feats)
        else:
            raise TypeError
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # N x K
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0):  # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )

        # 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def bag_forward(self, feats, c):
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        # sort class scores along the instance dimension, m_indices in shape N x C
        _, m_indices = torch.sort(c, 0, descending=True)
        # select critical instances, m_feats in shape C x K
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = torch.mm(Q, q_max.transpose(0, 1))
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)  # N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V
        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        # C = self.fcc(B)  # 1 x C x 1
        return B, B.detach()

    def batch_forward(self, feats, c):
        B_list, B_detach_list = [], []
        for f, cc in zip(feats, c):
            B, B_detach = self.bag_forward(f, cc)
            B_list.append(B)
            B_detach_list.append(B_detach)
        B = torch.cat(B_list)
        B_detach = torch.cat(B_detach_list)
        return B, B_detach

    def forward(self, feats, c):  # N x K, N x C
        if isinstance(feats, torch.Tensor) and isinstance(c, torch.Tensor):
            B, B_detach = self.bag_forward(feats, c)
        elif isinstance(feats, list) and isinstance(c, list):
            B, B_detach = self.batch_forward(feats, c)
        else:
            raise TypeError
        return B, B_detach


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, prediction_bag_detach = self.b_classifier(feats, classes)

        return classes, prediction_bag, prediction_bag_detach


def build_dsmil(dim_feat, num_classes):
    i_classifier = FCLayer(in_size=dim_feat, out_size=num_classes).cuda()
    b_classifier = BClassifier(input_size=dim_feat, output_class=num_classes).cuda()
    return MILNet(i_classifier, b_classifier).cuda()
