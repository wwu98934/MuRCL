import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.hidden = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.hidden[:]


class ActorCritic(nn.Module):
    def __init__(self, feature_dim, state_dim, hidden_state_dim=1024, policy_conv=False, action_std=0.1, action_size=2):
        super(ActorCritic, self).__init__()

        # encoder with convolution layer for MobileNetV3, EfficientNet and RegNet
        if policy_conv:
            self.state_encoder = nn.Sequential(
                nn.Conv2d(feature_dim, 32, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(int(state_dim * 32 / feature_dim), hidden_state_dim),
                nn.ReLU()
            )
        # encoder with linear layer for ResNet and DenseNet
        else:
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, hidden_state_dim),
                nn.ReLU()
            )

        self.gru = nn.GRU(hidden_state_dim, hidden_state_dim, batch_first=False)

        self.actor = nn.Sequential(
            nn.Linear(hidden_state_dim, action_size),
            nn.Sigmoid())

        self.critic = nn.Sequential(
            nn.Linear(hidden_state_dim, 1))

        self.action_var = torch.full((action_size,), action_std).cuda()

        self.hidden_state_dim = hidden_state_dim
        self.policy_conv = policy_conv
        self.feature_dim = feature_dim
        self.feature_ratio = int(math.sqrt(state_dim / feature_dim))

    def forward(self):
        raise NotImplementedError

    def act(self, state_ini, memory, restart_batch=False, training=False):
        if restart_batch:
            del memory.hidden[:]
            memory.hidden.append(torch.zeros(1, state_ini.size(0), self.hidden_state_dim).cuda())

        if not self.policy_conv:
            state = state_ini.flatten(1)
        else:
            state = state_ini

        state = self.state_encoder(state)

        state, hidden_output = self.gru(state.view(1, state.size(0), state.size(1)), memory.hidden[-1])
        memory.hidden.append(hidden_output)

        state = state[0]
        action_mean = self.actor(state)

        cov_mat = torch.diag(self.action_var).cuda()
        dist = torch.distributions.multivariate_normal.MultivariateNormal(action_mean, scale_tril=cov_mat)
        action = dist.sample().cuda()
        if training:
            action = F.relu(action)
            action = 1 - F.relu(1 - action)
            action_logprob = dist.log_prob(action).cuda()
            memory.states.append(state_ini)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
        else:
            action = action_mean

        return action.detach()

    def evaluate(self, state, action):
        seq_l = state.size(0)
        batch_size = state.size(1)

        if not self.policy_conv:
            state = state.flatten(2)
            state = state.view(seq_l * batch_size, state.size(2))
        else:
            state = state.view(seq_l * batch_size, state.size(2), state.size(3), state.size(4))

        state = self.state_encoder(state)
        state = state.view(seq_l, batch_size, -1)

        state, hidden = self.gru(state, torch.zeros(1, batch_size, state.size(2)).cuda())
        state = state.view(seq_l * batch_size, -1)

        action_mean = self.actor(state)

        cov_mat = torch.diag(self.action_var).cuda()

        dist = torch.distributions.multivariate_normal.MultivariateNormal(action_mean, scale_tril=cov_mat)

        action_logprobs = dist.log_prob(torch.squeeze(action.view(seq_l * batch_size, -1))).cuda()
        dist_entropy = dist.entropy().cuda()
        state_value = self.critic(state)

        return action_logprobs.view(seq_l, batch_size), \
               state_value.view(seq_l, batch_size), \
               dist_entropy.view(seq_l, batch_size)


class PPO:
    def __init__(self, feature_dim, state_dim, hidden_state_dim, policy_conv,
                 action_std=0.1, lr=0.0003, betas=(0.9, 0.999), gamma=0.7, K_epochs=1, eps_clip=0.2, action_size=2):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(feature_dim, state_dim, hidden_state_dim, policy_conv, action_std, action_size).cuda()

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(feature_dim, state_dim, hidden_state_dim, policy_conv, action_std,
                                      action_size).cuda()
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state, memory, restart_batch=False, training=True):
        return self.policy_old.act(state, memory, restart_batch, training)

    def update(self, memory):
        rewards = []
        discounted_reward = 0

        for reward in reversed(memory.rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        rewards = torch.cat(rewards, 0).cuda()

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        old_states = torch.stack(memory.states, 0).cuda().detach()
        old_actions = torch.stack(memory.actions, 0).cuda().detach()
        old_logprobs = torch.stack(memory.logprobs, 0).cuda().detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # print(f"logprobs: {logprobs.shape}, state_values: {state_values.shape}, dist_entropy:{dist_entropy.shape}")
            # print(f"rewards: {rewards.shape}")
            ratios = torch.exp(logprobs - old_logprobs.detach())

            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())


class Full_layer(torch.nn.Module):
    def __init__(self, feature_num, hidden_state_dim=1024, fc_rnn=True, class_num=1000):
        super(Full_layer, self).__init__()
        self.class_num = class_num
        self.feature_num = feature_num

        self.hidden_state_dim = hidden_state_dim
        self.hidden = None
        self.fc_rnn = fc_rnn

        # classifier with RNN for ResNet, DenseNet and RegNet
        if fc_rnn:
            self.rnn = nn.GRU(feature_num, self.hidden_state_dim)
            self.fc = nn.Linear(self.hidden_state_dim, class_num)
        # cascaded classifier for MobileNetV3 and EfficientNet
        else:
            self.fc_2 = nn.Linear(self.feature_num * 2, class_num)
            self.fc_3 = nn.Linear(self.feature_num * 3, class_num)
            self.fc_4 = nn.Linear(self.feature_num * 4, class_num)
            self.fc_5 = nn.Linear(self.feature_num * 5, class_num)

    def forward(self, x, restart=False):

        if self.fc_rnn:
            if restart:
                # print(f"In Fully_layer restart x: {x.shape}")
                output, h_n = self.rnn(x.view(1, x.size(0), x.size(1)),
                                       torch.zeros(1, x.size(0), self.hidden_state_dim).cuda())
                self.hidden = h_n
            else:
                output, h_n = self.rnn(x.view(1, x.size(0), x.size(1)), self.hidden)
                self.hidden = h_n

            return self.fc(output[0])
        else:
            if restart:
                self.hidden = x
            else:
                self.hidden = torch.cat([self.hidden, x], 1)

            if self.hidden.size(1) == self.feature_num:
                return None
            elif self.hidden.size(1) == self.feature_num * 2:
                return self.fc_2(self.hidden)
            elif self.hidden.size(1) == self.feature_num * 3:
                return self.fc_3(self.hidden)
            elif self.hidden.size(1) == self.feature_num * 4:
                return self.fc_4(self.hidden)
            elif self.hidden.size(1) == self.feature_num * 5:
                return self.fc_5(self.hidden)
            else:
                print(self.hidden.size())
                exit()


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

    def forward(self, x):  # B x N x dim_in, a bag
        # print(f"input x: {x.shape}")
        if isinstance(x, list):
            x = [t.squeeze(0) for t in x]
            H = [self.encoder(t) for t in x]
            A = [self.attention(h) for h in H]  # N x K
            A = [torch.transpose(a, -2, -1) for a in A]  # K x N
            A = [a / math.sqrt(a.shape[-1]) for a in A]
            A = [F.softmax(a, dim=-1) for a in A]  # softmax over N
            M = torch.cat([torch.mm(A[i], H[i]) for i in range(len(H))])  # K x N * N x L -> K x L

        elif isinstance(x, torch.Tensor):
            H = self.encoder(x)  # NxL
            A = self.attention(H)  # N x K
            # print(f"A: {A.shape}")
            A = torch.transpose(A, -2, -1)  # K x N
            # print(f"A: {A.shape}")
            A = A / math.sqrt(A.shape[-1])
            # print(f"A: {A.shape}\n{A}")
            A = F.softmax(A, dim=-1)  # softmax over N
            M = torch.cat([torch.mm(A[i], H[i]) for i in range(H.shape[0])])  # K x N * N x L -> K x L
        else:
            raise TypeError
        # print(f"{H.shape}")
        # print(f"H: {H.shape}")
        # H = self.encoder(x)  # NxL
        # print(f"H: {H.shape}")

        # A = self.attention(H)  # N x K
        # # print(f"A: {A.shape}")
        # A = torch.transpose(A, -2, -1)  # K x N
        # # print(f"A: {A.shape}")
        # A = A / math.sqrt(A.shape[-1])
        # A = F.softmax(A, dim=-1)  # softmax over N
        # # print(f"A: {A.shape}")
        # M = torch.cat([torch.mm(A[i], H[i]) for i in range(H.shape[0])])  # K x N * N x L -> K x L
        # print(f"M: {M.shape}")

        x = self.decoder(M)
        # print(f"x: {x.shape}")
        # x = self.fc(x)
        # print(f"x: {x.shape}")

        return x, x.detach()


class ABMIL_S(nn.Module):
    def __init__(self, dim_in, L=512, D=128, K=1, dim_out=2):
        super(ABMIL_S, self).__init__()
        self.L, self.D, self.K = L, D, K

        self.encoder = nn.Sequential(
            nn.Linear(dim_in, self.L),
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

    def forward(self, x):  # B x N x dim_in, a bag
        # x = x.squeeze(0)
        if isinstance(x, list):
            x = [t.squeeze(0) for t in x]
            # for t in x:
            #     print(f"t: {t.shape}")
            H = [self.encoder(t) for t in x]
            A = [self.attention(h) for h in H]  # N x K
            # print(f"A: {A.shape}")
            A = [torch.transpose(a, -2, -1) for a in A]  # K x N
            # print(f"A: {A.shape}")
            A = [a / math.sqrt(a.shape[-1]) for a in A]
            A = [F.softmax(a, dim=-1) for a in A]  # softmax over N
            # print(f"A: {A.shape}")
            # for h in H:
            #     print(f"h:{h.shape}")
            # for a in A:
            #     print(f"a:{a.shape}")
            M = torch.cat([torch.mm(A[i], H[i]) for i in range(len(H))])  # K x N * N x L -> K x L

        elif isinstance(x, torch.Tensor):
            H = self.encoder(x)  # NxL
            A = self.attention(H)  # N x K
            # print(f"A: {A.shape}")
            A = torch.transpose(A, -2, -1)  # K x N
            # print(f"A: {A.shape}")
            A = A / math.sqrt(A.shape[-1])
            A = F.softmax(A, dim=-1)  # softmax over N
            # print(f"A: {A.shape}")
            M = torch.cat([torch.mm(A[i], H[i]) for i in range(H.shape[0])])  # K x N * N x L -> K x L
            # print(f"M: {M.shape}")

            # x = self.decoder(M)
            # print(f"x: {x.shape}")
            # x = self.fc(x)
        else:
            raise TypeError
        x = self.decoder(M)
        # print(f"x: {x.shape}")

        return x, x.detach()


class CL(nn.Module):
    def __init__(self, encoder, projection_dim, n_features):
        super(CL, self).__init__()
        # assert hasattr(encoder, 'fc')
        self.encoder = encoder
        # self.encoder.fc = nn.Identity()

        self.projection_dim = projection_dim
        self.n_features = n_features

        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_views):
        assert isinstance(x_views, list), f""
        h_views = [self.encoder(x)[0] for x in x_views]
        # z_views = [self.projector(h) for h in h_views]
        return h_views, [h.detach() for h in h_views]


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
        # assert len(feats.shape) == 3 and feats.shape[0] == 1, f"feats.shape: {feats.shape}"
        # feats = feats.squeeze(0)
        # print(f"{feats.shape}")
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0,
                                  descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0,
                                     index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0,
                                        1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)  # N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V
        # print(f"B: {B.shape}")
        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        # C = self.fcc(B)  # 1 x C x 1
        # C = C.view(1, -1)
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


# START: CLAM ----------------------------------------------------------------------------------------------------------
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net(nn.Module):

    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x


class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, in_dim=512):
        super(CLAM_SB, self).__init__()
        self.size_dict = {"small": [in_dim, 512, 256], "big": [in_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping

        initialize_weights(self)

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length,), 1, device=device).long()

    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length,), 0, device=device).long()

    # instance-level evaluation for in-the-class attention branch
    def inst_eval(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        top_n_ids = torch.topk(-A, self.k_sample, dim=1)[1][-1]
        top_n = torch.index_select(h, dim=0, index=top_n_ids)
        p_targets = self.create_positive_targets(self.k_sample, device)
        n_targets = self.create_negative_targets(self.k_sample, device)

        all_targets = torch.cat([p_targets, n_targets], dim=0)
        all_instances = torch.cat([top_p, top_n], dim=0)
        logits = classifier(all_instances)
        all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, all_targets)
        return instance_loss, all_preds, all_targets

    # instance-level evaluation for out-of-the-class attention branch
    def inst_eval_out(self, A, h, classifier):
        device = h.device
        if len(A.shape) == 1:
            A = A.view(1, -1)
        top_p_ids = torch.topk(A, self.k_sample)[1][-1]
        top_p = torch.index_select(h, dim=0, index=top_p_ids)
        p_targets = self.create_negative_targets(self.k_sample, device)
        logits = classifier(top_p)
        p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
        instance_loss = self.instance_loss_fn(logits, p_targets)
        return instance_loss, p_preds, p_targets

    def bag_forward(self, bag, label=None, instance_eval=False, return_features=False, attention_only=False):
        # device = h.device
        if len(bag.shape) == 3 and bag.shape[0] == 1:
            bag = bag.squeeze(0)
        assert len(bag.shape) == 2, f"h.shape: {bag.shape}"
        A, h = self.attention_net(bag)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        # A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A, h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        # logits = self.classifiers(M)
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return M, results_dict

    def batch_forward(self, batch, label=None, instance_eval=False, return_features=False, attention_only=False):
        outputs, results_dict_batch = [], []
        if label is None:
            for bag in batch:
                output, results_dict = self.bag_forward(bag, label, instance_eval, return_features, attention_only)
                outputs.append(output)
                results_dict_batch.append(results_dict)
        else:
            for bag, lbl in zip(batch, label):
                output, results_dict = self.bag_forward(bag, lbl, instance_eval, return_features, attention_only)
                outputs.append(output)
                results_dict_batch.append(results_dict)
        return torch.cat(outputs, 0), results_dict_batch

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        if isinstance(h, list):
            outputs, results_dict = self.batch_forward(h, label, instance_eval, return_features, attention_only)
        elif isinstance(h, torch.Tensor):
            if h.shape[0] == 1:
                outputs, results_dict = self.bag_forward(h.squeeze(0), label, instance_eval, return_features,
                                                         attention_only)
            else:
                outputs, results_dict = self.batch_forward(h, label, instance_eval, return_features, attention_only)
        else:
            raise TypeError
        if instance_eval:
            return outputs, outputs.detach(), results_dict
        else:
            return outputs, outputs.detach()


class CLAM_MB(CLAM_SB):
    def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=2,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, in_dim=512):
        nn.Module.__init__(self)
        self.size_dict = {"small": [in_dim, 512, 256], "big": [in_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=n_classes)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        bag_classifiers = [nn.Linear(size[1], 1) for i in
                           range(n_classes)]  # use an indepdent linear layer to predict each class
        self.classifiers = nn.ModuleList(bag_classifiers)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping
        initialize_weights(self)

    def bag_forward(self, bag, label=None, instance_eval=False, return_features=False, attention_only=False):
        if len(bag.shape) == 3 and bag.shape[0] == 1:
            bag = bag.squeeze(0)
        assert len(bag.shape) == 2, f"h.shape: {bag.shape}"
        device = bag.device
        A, h = self.attention_net(bag)  # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        if attention_only:
            return A
        # A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        if instance_eval:
            total_inst_loss = 0.0
            all_preds = []
            all_targets = []
            inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
            for i in range(len(self.instance_classifiers)):
                inst_label = inst_labels[i].item()
                classifier = self.instance_classifiers[i]
                if inst_label == 1:  # in-the-class:
                    instance_loss, preds, targets = self.inst_eval(A[i], h, classifier)
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                else:  # out-of-the-class
                    if self.subtyping:
                        instance_loss, preds, targets = self.inst_eval_out(A[i], h, classifier)
                        all_preds.extend(preds.cpu().numpy())
                        all_targets.extend(targets.cpu().numpy())
                    else:
                        continue
                total_inst_loss += instance_loss

            if self.subtyping:
                total_inst_loss /= len(self.instance_classifiers)

        M = torch.mm(A, h)
        logits = torch.empty(1, self.n_classes).float().to(device)
        for c in range(self.n_classes):
            logits[0, c] = self.classifiers[c](M[c])
        # Y_hat = torch.topk(logits, 1, dim=1)[1]
        # Y_prob = F.softmax(logits, dim=1)
        if instance_eval:
            results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
                            'inst_preds': np.array(all_preds)}
        else:
            results_dict = {}
        if return_features:
            results_dict.update({'features': M})
        return logits, results_dict

    def batch_forward(self, batch, label=None, instance_eval=False, return_features=False, attention_only=False):
        outputs, results_dict_batch = [], []
        for bag, lbl in zip(batch, label):
            output, results_dict = self.bag_forward(bag, lbl, instance_eval, return_features, attention_only)
            outputs.append(output)
            results_dict_batch.append(results_dict)
        return torch.cat(outputs, 0), results_dict_batch

    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        if isinstance(h, list):
            outputs, results_dict = self.batch_forward(h, label, instance_eval, return_features, attention_only)
        elif isinstance(h, torch.Tensor):
            if h.shape[0] == 1:
                outputs, results_dict = self.bag_forward(h.squeeze(0), label, instance_eval, return_features,
                                                         attention_only)
            else:
                outputs, results_dict = self.batch_forward(h, label, instance_eval, return_features, attention_only)
        else:
            raise TypeError
        return outputs, outputs.detach(), results_dict


# END: CLAM ------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # model = ABMIL(1024)
    # inputs = [torch.randn((1, 16, 1024)), torch.randn((1, 12, 1024)), torch.randn((1, 9, 1024)),
    #           torch.randn((1, 21, 1024))]
    # x_, x_detach = model(inputs)
    # print(x_.shape, '\n', x_detach.shape)
    # inputs = torch.randn((3, 3, 1024))
    # x_, x_detach = model(inputs)
    # print(x_.shape, '\n', x_detach.shape)
    #
    # dsmil = build_dsmil(dim_feat=512, num_classes=2)
    # inputs_bag = torch.randn((1, 1245, 512))
    # inputs_batch = [torch.randn((1, 16, 512)), torch.randn((1, 12, 512)), torch.randn((1, 9, 512)),
    #                 torch.randn((1, 21, 512))]
    # inputs_batch = torch.randn((16, 1024, 512))
    # classes, prediction_bag, prediction_bag_detach = dsmil(inputs_batch)
    # print('-------------------------------')
    # # print(classes.shape, prediction_bag.shape, prediction_bag_detach.shape)
    # print(prediction_bag.shape, prediction_bag_detach.shape)
    # for x in classes:
    #     print(x.shape)

    clam = CLAM_SB()
    # inputs = [torch.randn((16, 512)), torch.randn((12, 512)), torch.randn((9, 512)), torch.randn((21, 512))]
    inputs = torch.randn((1, 16, 512))
    labels = torch.tensor([1])
    out_, out_detach, res_dict = clam(inputs, label=labels, instance_eval=True)
    print(out_.shape)
    print(res_dict)
    ...
