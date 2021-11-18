import os
import copy
import shutil
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import nni
from nni.utils import merge_parameter

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import WSIWithCluster
from utils.general import AverageMeter, CSVWriter, EarlyStop, increment_path, BestVariable, accuracy, init_seeds, \
    load_json, get_metrics, get_score
from models import rlmil


def get_feats(feat_list, clusters_list, action_sequence, feat_size=256):
    """Get small patch of the original image"""
    batch_size = len(feat_list)
    device = action_sequence.device

    feats = []
    for i in range(batch_size):
        num_patch = feat_list[i].shape[-2]
        sample_ratio = feat_size / num_patch
        num_feats_cluster = torch.tensor([len(c) for c in clusters_list[i]], device=device)
        num_feats_cluster_size = torch.round(num_feats_cluster * sample_ratio).int()
        feat_coordinate_l = torch.floor(action_sequence[i] * (num_feats_cluster - num_feats_cluster_size)).int()
        feat_coordinate_r = feat_coordinate_l + num_feats_cluster_size
        indices = []
        for j, c in enumerate(clusters_list[i]):
            index = c[feat_coordinate_l[j].item():feat_coordinate_r[j].item()]
            indices.extend(index)
        indices = sorted(indices)
        per_feat = feat_list[i][:, indices, :]
        if per_feat.shape[-2] < feat_size:
            margin = feat_size - per_feat.shape[-2]
            feat_pad = torch.zeros(size=(1, margin, per_feat.shape[-1]), device=device)
            per_feat = torch.cat((per_feat, feat_pad), dim=1)
        else:
            per_feat = per_feat[:, :feat_size, :]
        feats.append(per_feat)
    feats = torch.cat(feats, 0)
    return feats


def mixup(inputs, alpha):
    batch_size = inputs.shape[0]
    lambda_ = alpha + torch.rand(size=(batch_size, 1), device=inputs.device) * (1 - alpha)
    rand_idx = torch.randperm(batch_size, device=inputs.device)
    a = torch.stack([lambda_[i] * inputs[i] for i in range(batch_size)])
    b = torch.stack([(1 - lambda_[i]) * inputs[rand_idx[i]] for i in range(batch_size)])
    outputs = a + b
    return outputs, lambda_, rand_idx


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def create_save_dir(args):
    dir1 = f'{args.dataset}_np_{args.feat_size}'
    dir2 = f'RLMIL'
    rlmil_setting = [
        f'T{args.T}',
        f'dr{args.dsn_ratio}',
        f'phd{args.policy_hidden_dim}',
        f'as{args.action_std}',
        f'pg{args.ppo_gamma}',
        f'ke{args.K_epochs}',
        f'fhd{args.fc_hidden_dim}',
    ]
    dir3 = '_'.join(rlmil_setting)
    dir4 = args.arch
    # Arch Setting
    if args.arch in ['ABMIL']:
        arch_setting = [
            f'L{args.L}',
            f'D{args.D}',
            f'dpt{args.dropout}',
        ]
    elif args.arch in ['DSMIL']:
        arch_setting = ['default']
    elif args.arch in ['CLAM_SB', 'CLAM_MB']:
        arch_setting = [
            f"size_{args.size_arg}",
            f"ks_{args.k_sample}",
            f"bw_{args.bag_weight}"
        ]
    else:
        raise ValueError()
    dir5 = '_'.join(arch_setting)
    dir6 = args.train_method
    dir7 = f'exp'
    if args.save_dir_flag is not None:
        dir7 = f'{dir7}_{args.save_dir_flag}'
    dir8 = f'seed{args.seed}'
    dir9 = f'stage_{args.train_stage}'
    if args.nni:
        args.save_dir = str(
            Path(args.base_save_dir) / 'nni_search' / dir1 / dir2 / dir3 / dir4 / dir5 / dir6 / dir7 / dir8 / dir9)
    else:
        args.save_dir = str(Path(args.base_save_dir) / dir1 / dir2 / dir3 / dir4 / dir5 / dir6 / dir7 / dir8 / dir9)
    print(f"save_dir: {args.save_dir}")


def get_datasets(args):
    print(f"train_data: {args.train_data}")
    indices = load_json(args.data_split_json)
    train_set = WSIWithCluster(
        args.data_csv,
        indices=indices[args.train_data],
        num_sample_patches=args.feat_size,
        shuffle=True,
        preload=args.preload
    )
    valid_set = WSIWithCluster(
        args.data_csv,
        indices=indices['valid'],
        num_sample_patches=args.feat_size,
        shuffle=False,
        preload=args.preload
    )
    test_set = WSIWithCluster(
        args.data_csv,
        indices=indices['test'],
        num_sample_patches=args.feat_size,
        shuffle=False,
        preload=args.preload
    )
    args.num_clusters = train_set.num_clusters
    return {'train': train_set, 'valid': valid_set, 'test': test_set}, train_set.patch_dim, len(train_set)


def create_model(args, dim_patch):
    print(f"Creating model {args.arch}...")
    if args.arch == 'ABMIL':
        model = rlmil.ABMIL(
            dim_in=dim_patch,
            L=args.L,
            D=args.D,
            dim_out=args.num_classes,
            dropout=args.dropout,
        )
        model_prime = rlmil.ABMIL(
            dim_in=dim_patch,
            L=args.L,
            D=args.D,
            dim_out=args.num_classes,
            dropout=args.dropout,
        )
        args.feature_num = args.L
    elif args.arch == 'DSMIL':
        model = rlmil.build_dsmil(dim_feat=dim_patch, num_classes=args.num_classes)
        model_prime = rlmil.build_dsmil(dim_feat=dim_patch, num_classes=args.num_classes)
        args.feature_num = dim_patch
    elif args.arch == 'CLAM_SB':
        model = rlmil.CLAM_SB(
            gate=True,
            size_arg=args.size_arg,
            dropout=True,
            k_sample=args.k_sample,
            n_classes=args.num_classes,
            subtyping=True,
            in_dim=dim_patch
        )
        model_prime = rlmil.CLAM_SB(
            gate=True,
            size_arg=args.size_arg,
            dropout=True,
            k_sample=args.k_sample,
            n_classes=args.num_classes,
            subtyping=True,
            in_dim=dim_patch
        )
        args.feature_num = dim_patch
    else:
        raise ValueError(f'args.arch error, {args.arch}. ')
    fc = rlmil.Full_layer(args.feature_num, args.fc_hidden_dim, args.fc_rnn, args.num_classes)
    ppo = None

    if args.train_method in ['finetune', 'linear']:
        if args.train_stage == 1:
            assert args.checkpoint is not None and Path(args.checkpoint).exists(), f"{args.checkpoint} is not exists!"

            checkpoint = torch.load(args.checkpoint)
            model_state_dict = checkpoint['model_state_dict']
            model_prime_state_dict = checkpoint['model_prime_state_dict']
            fc_state_dict = checkpoint['fc']
            for k in list(model_state_dict.keys()):
                print(f"key: {k}")
            for k in list(model_state_dict.keys()):
                if k.startswith('encoder') and not k.startswith('encoder.fc') and not k.startswith(
                        'encoder.classifiers'):
                    model_state_dict[k[len('encoder.'):]] = model_state_dict[k]
                del model_state_dict[k]
            for k in list(model_prime_state_dict.keys()):
                if k.startswith('encoder') and not k.startswith('encoder.fc') and not k.startswith(
                        'encoder.classifiers'):
                    model_prime_state_dict[k[len('encoder.'):]] = model_prime_state_dict[k]
                del model_prime_state_dict[k]
            for k in list(model_state_dict.keys()):
                print(f"key: {k}")
            msg_model = model.load_state_dict(model_state_dict, strict=False)
            msg_model_prime = model_prime.load_state_dict(model_prime_state_dict, strict=False)
            if args.load_fc:
                msg_fc = fc.load_state_dict(fc_state_dict, strict=False)
                print(f"msg_fc missing_keys: {msg_fc.missing_keys}")
            print(f"msg_model missing_keys: {msg_model.missing_keys}")
            print(f"msg_model_prime missing_keys: {msg_model_prime.missing_keys}")

            if args.train_method == 'linear':
                for n, p in model.named_parameters():
                    # print(f"key: {n}")
                    if n.startswith('fc') or n.startswith('classifiers') or n.startswith('instance_classifiers'):
                        print(f"not_fixed_key: {n}")
                    else:
                        p.requires_grad = False
                for n, p in model_prime.named_parameters():
                    # print(f"key: {n}")
                    if n.startswith('fc') or n.startswith('classifiers') or n.startswith('instance_classifiers'):
                        print(f"not_fixed_key: {n}")
                    else:
                        p.requires_grad = False

            if args.arch == 'ABMIL':
                dim_in = model.fc.in_features
                model.fc = torch.nn.Linear(dim_in, args.num_classes)
                dim_in = model_prime.fc.in_features
                model_prime.fc = torch.nn.Linear(dim_in, args.num_classes)
            elif args.arch == 'CLAM_SB':
                dim_in = model.classifiers.in_features
                model.classifiers = torch.nn.Linear(dim_in, args.num_classes)
                dim_in = model_prime.classifiers.in_features
                model_prime.classifiers = torch.nn.Linear(dim_in, args.num_classes)
            else:
                raise NotImplementedError

        elif args.train_stage == 2:
            if args.checkpoint_path is None:
                args.checkpoint_path = str(Path(args.save_dir).parent / 'stage_1' / 'model_best.pth.tar')
            assert Path(args.checkpoint_path).exists(), f"{args.checkpoint_path} is not exist!"
            # assert Path(args.checkpoint_path).parent.parent == Path(args.save_dir).parent
            assert Path(args.checkpoint_path).parent.stem == 'stage_1'

            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_prime.load_state_dict(checkpoint['model_prime_state_dict'])
            fc.load_state_dict(checkpoint['fc'])

            assert args.checkpoint is not None and Path(args.checkpoint).exists(), f"{args.checkpoint} is not exists!"
            checkpoint = torch.load(args.checkpoint)
            state_dim = args.L
            ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                            action_std=args.action_std,
                            lr=args.ppo_lr,
                            gamma=args.ppo_gamma,
                            K_epochs=args.K_epochs,
                            action_size=args.num_clusters)
            ppo.policy.load_state_dict(checkpoint['policy'])
            ppo.policy_old.load_state_dict(checkpoint['policy'])

        elif args.train_stage == 3:
            if args.checkpoint_path is None:
                args.checkpoint_path = str(Path(args.save_dir).parent / 'stage_2' / 'model_best.pth.tar')
            assert Path(args.checkpoint_path).exists(), f"{args.checkpoint_path} is not exist!"
            # assert Path(args.checkpoint_path).parent.parent == Path(args.save_dir).parent
            assert Path(args.checkpoint_path).parent.stem == 'stage_2'

            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_prime.load_state_dict(checkpoint['model_prime_state_dict'])
            fc.load_state_dict(checkpoint['fc'])

            state_dim = args.L
            ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                            action_std=args.action_std,
                            lr=args.ppo_lr,
                            gamma=args.ppo_gamma,
                            K_epochs=args.K_epochs,
                            action_size=args.num_clusters)
            ppo.policy.load_state_dict(checkpoint['policy'])
            ppo.policy_old.load_state_dict(checkpoint['policy'])

            if args.train_method == 'linear':
                for n, p in model.named_parameters():
                    # print(f"key: {n}")
                    if n.startswith('fc') or n.startswith('classifiers') or n.startswith('instance_classifiers'):
                        print(f"not_fixed_key: {n}")
                    else:
                        p.requires_grad = False
                for n, p in model_prime.named_parameters():
                    # print(f"key: {n}")
                    if n.startswith('fc') or n.startswith('classifiers') or n.startswith('instance_classifiers'):
                        print(f"not_fixed_key: {n}")
                    else:
                        p.requires_grad = False

            # dim_in = model.fc.in_features
            # model.fc = torch.nn.Linear(dim_in, args.num_classes)
            # dim_in = model_prime.fc.in_features
            # model_prime.fc = torch.nn.Linear(dim_in, args.num_classes)
        else:
            raise ValueError
    elif args.train_method in ['scratch']:
        if args.train_stage == 1:
            pass
        elif args.train_stage == 2:
            if args.checkpoint_path is None:
                args.checkpoint_path = str(Path(args.save_dir).parent / 'stage_1' / 'model_best.pth.tar')
            assert Path(args.checkpoint_path).exists(), f"{args.checkpoint_path} is not exist!"
            # assert Path(args.checkpoint_path).parent.parent == Path(
            #     args.save_dir).parent, f"checkpoint_path:\n{args.checkpoint_path}\nsave_dir:\n{args.save_dir}"
            assert Path(args.checkpoint_path).parent.stem == 'stage_1'

            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_prime.load_state_dict(checkpoint['model_prime_state_dict'])
            fc.load_state_dict(checkpoint['fc'])

            state_dim = args.L
            ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                            action_std=args.action_std,
                            lr=args.ppo_lr,
                            gamma=args.ppo_gamma,
                            K_epochs=args.K_epochs,
                            action_size=args.num_clusters)
        elif args.train_stage == 3:
            if args.checkpoint_path is None:
                args.checkpoint_path = str(Path(args.save_dir).parent / 'stage_2' / 'model_best.pth.tar')
            assert Path(args.checkpoint_path).exists(), f'{str(args.checkpoint_path)} is not exists!'
            # assert Path(args.checkpoint_path).parent.parent == Path(
            #     args.save_dir).parent, f"checkpoint_path:\n{args.checkpoint_path}\nsave_dir:\n{args.save_dir}"
            assert Path(args.checkpoint_path).parent.stem == 'stage_2'

            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model_prime.load_state_dict(checkpoint['model_prime_state_dict'])
            fc.load_state_dict(checkpoint['fc'])

            state_dim = args.L
            ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                            action_std=args.action_std,
                            lr=args.ppo_lr,
                            gamma=args.ppo_gamma,
                            K_epochs=args.K_epochs,
                            action_size=args.num_clusters)
            ppo.policy.load_state_dict(checkpoint['policy'])
            ppo.policy_old.load_state_dict(checkpoint['policy'])
        else:
            raise ValueError
    else:
        raise ValueError

    model = torch.nn.DataParallel(model).cuda()
    model_prime = torch.nn.DataParallel(model_prime).cuda()
    fc = fc.cuda()

    assert model is not None, "creating model failed. "
    print(f"model Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"model_prime Total params: {sum(p.numel() for p in model_prime.parameters()) / 1e6:.2f}M")
    print(f"fc Total params: {sum(p.numel() for p in fc.parameters()) / 1e6:.2f}M")
    return model, model_prime, fc, ppo


def get_criterion(args):
    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError(f"args.loss error, error value is {args.loss}.")
    return criterion


def get_optimizer(args, model, model_prime, fc):
    if args.train_stage != 2:
        if args.train_model_prime:
            params = [{'params': model.parameters(), 'lr': args.backbone_lr},
                      {'params': model_prime.parameters(), 'lr': args.backbone_lr},
                      {'params': fc.parameters(), 'lr': args.fc_lr}]
        else:
            params = [{'params': model.parameters(), 'lr': args.backbone_lr},
                      {'params': fc.parameters(), 'lr': args.fc_lr}]
        if args.optimizer == 'SGD':
            optimizer = torch.optim.SGD(params,
                                        lr=0,  # specify in params
                                        momentum=args.momentum,
                                        nesterov=args.nesterov,
                                        weight_decay=args.wdecay)
        elif args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(params, betas=(args.beta1, args.beta2), weight_decay=args.wdecay)
        else:
            raise NotImplementedError
    else:
        optimizer = None
        args.epochs = 10
    return optimizer


def get_scheduler(args, optimizer):
    if optimizer is None:
        return None
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup, eta_min=1e-6)
    elif args.scheduler is None:
        scheduler = None
    else:
        raise ValueError("Optimizer not found. Accepted 'Adam', 'SGD' or 'RMSprop'")
    return scheduler


# Train Model Functions ------------------------------------------------------------------------------------------------

def train_CLAM(args, epoch, train_set, model, model_prime, fc, ppo, memory, criterion, optimizer, scheduler):
    """
    一个epoch
    :param args:
    :param epoch:
    :param train_set:
    :param model: local
    :param model_prime:
    :param fc:
    :param ppo:
    :param memory:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :return:
    """
    print(f"train_CLAM...")
    length = len(train_set)
    print(f"data length: {length}")
    train_set.shuffle()

    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    if args.train_stage == 2:
        model_prime.eval()
        model.eval()
        fc.eval()
    else:
        if args.train_model_prime:
            model_prime.train()
        else:
            model_prime.eval()
        model.train()
        fc.train()

    dsn_fc_prime = model_prime.module.classifiers
    # print(f"dsn_fc_prime: {dsn_fc_prime}")
    dsn_fc = model.module.classifiers
    # print(f"dsn_fc: {dsn_fc}")

    progress_bar = tqdm(range(args.num_data))
    feat_list, cluster_list, label_list, step = [], [], [], 0
    batch_idx = 0
    labels_list, outputs_list = [], []
    for data_idx in progress_bar:
        loss_cla = []
        loss_list_dsn = []

        feat, cluster, label, case_id = train_set[data_idx % length]
        assert len(feat.shape) == 2, f"{feat.shape}"
        feat = feat.unsqueeze(0).to(args.device)
        label = label.unsqueeze(0).to(args.device)

        feat_list.append(feat)
        cluster_list.append(cluster)
        label_list.append(label)

        step += 1
        if step == args.batch_size or data_idx == args.num_data - 1:
            labels = torch.cat(label_list)
            action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
            feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
            if args.mix_up:
                feats = mixup(feats, args.alpha)[0]
            if args.train_model_prime and args.train_stage != 2:
                outputs, states, result_dict = model_prime(feats, label=labels, instance_eval=True)
                outputs_dsn = dsn_fc_prime(outputs)
                outputs = fc(outputs, restart=True)
            else:
                with torch.no_grad():
                    outputs, states, result_dict = model_prime(feats, label=labels, instance_eval=True)
                    outputs_dsn = dsn_fc_prime(outputs)
                    outputs = fc(outputs, restart=True)

            loss_prime = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * result_dict[
                'instance_loss']
            # print(f"loss_prime: {loss_prime}")
            loss_cla.append(loss_prime)

            loss_dsn = args.bag_weight * criterion(outputs_dsn, labels) + (1 - args.bag_weight) * result_dict[
                'instance_loss']
            loss_list_dsn.append(loss_dsn)

            losses[0].update(loss_prime.data.item(), outputs.shape[0])
            acc = accuracy(outputs, labels, topk=(1,))[0]
            top1[0].update(acc.item())

            # RL
            confidence_last = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
            for patch_step in range(1, args.T):
                if args.train_stage == 1:
                    action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
                else:
                    if patch_step == 1:
                        action_sequence = ppo.select_action(states.to(0), memory, restart_batch=True)
                    else:
                        action_sequence = ppo.select_action(states.to(0), memory)

                feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
                if args.mix_up:
                    feats = mixup(feats, args.alpha)[0]

                if args.train_stage != 2:
                    outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
                    outputs_dsn = dsn_fc(outputs)
                    outputs = fc(outputs, restart=False)
                else:
                    with torch.no_grad():
                        outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
                        outputs_dsn = dsn_fc(outputs)
                        outputs = fc(outputs, restart=False)
                loss = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * result_dict[
                    'instance_loss']
                loss_cla.append(loss)
                losses[patch_step].update(loss.data.item(), len(feat_list))

                loss_dsn = args.bag_weight * criterion(outputs_dsn, labels) + (1 - args.bag_weight) * result_dict[
                    'instance_loss']
                loss_list_dsn.append(loss_dsn)

                acc = accuracy(outputs, labels, topk=(1,))[0]
                top1[patch_step].update(acc.item())

                confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1,
                                                                                                                -1)
                reward = confidence - confidence_last
                confidence_last = confidence

                reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                memory.rewards.append(reward)

            # print(f"loss_cla:\n{loss_cla}")
            # print(f"loss_list_dsn:\n{loss_list_dsn}")
            loss = (sum(loss_cla) + args.dsn_ratio * sum(loss_list_dsn)) / args.T
            if args.train_stage != 2:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                ppo.update(memory)
            memory.clear_memory()
            torch.cuda.empty_cache()

            labels_list.append(labels.detach())
            outputs_list.append(outputs.detach())

            feat_list, cluster_list, label_list, step = [], [], [], 0
            batch_idx += 1
            progress_bar.set_description(
                f"Train Epoch: {epoch + 1:2}/{args.epochs:2}. Iter: {batch_idx:3}/{args.eval_step:3}. "
                f"Loss: {losses[-1].avg:.4f}. Acc: {top1[-1].avg:.4f}"
            )
            progress_bar.update()

            # _acc = [acc.avg for acc in top1]
            # print('accuracy of each step:')
            # print(_acc)

            # _reward = [reward.avg for reward in reward_list]
            # print('reward of each step:')
            # print(_reward)

    progress_bar.close()
    if args.train_stage != 2 and scheduler is not None and epoch >= args.warmup:
        scheduler.step()

    labels = torch.cat(labels_list)
    outputs = torch.cat(outputs_list)
    acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)
    # predict = torch.softmax(outputs, dim=1)
    # auc = roc_auc_score(labels.cpu().numpy(), predict[:, 1].cpu().numpy())

    return losses[-1].avg, acc, auc, precision, recall, f1_score


def test_CLAM(args, test_set, model, model_prime, fc, ppo, memory, criterion, mode):
    losses = [AverageMeter() for _ in range(args.T)]
    # top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]
    model_prime.eval()
    model.eval()
    fc.eval()
    dsn_fc_prime = model_prime.module.classifiers
    dsn_fc = model.module.classifiers
    # acc_list, auc_list = [], []
    with torch.no_grad():

        # progress_bar = tqdm(test_set)
        feat_list, cluster_list, label_list, case_id_list, step = [], [], [], [], 0
        # batch_idx = 0
        # labels_list, outputs_list = [], []
        for data_idx, (feat, cluster, label, case_id) in enumerate(test_set):
            loss_cla = []
            loss_list_dsn = []

            feat = feat.unsqueeze(0).to(args.device)
            label = label.unsqueeze(0).to(args.device)
            feat_list.append(feat)
            cluster_list.append(cluster)
            label_list.append(label)
            case_id_list.append(case_id)

        labels = torch.cat(label_list)
        action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
        feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
        # if args.mix_up:
        #     feats = mixup(feats, args.alpha)[0]

        outputs, states, result_dict = model_prime(feats, label=labels, instance_eval=True)
        outputs_dsn = dsn_fc_prime(outputs)
        outputs = fc(outputs, restart=True)

        ins_loss = 0
        for r in result_dict:
            ins_loss = ins_loss + r['instance_loss']
        ins_loss = ins_loss / len(feat_list)
        loss_prime = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * ins_loss
        loss_cla.append(loss_prime)

        loss_dsn = args.bag_weight * criterion(outputs_dsn, labels) + (1 - args.bag_weight) * ins_loss
        loss_list_dsn.append(loss_dsn)

        losses[0].update(loss_prime.data.item(), outputs.shape[0])
        # acc, auc = acc_auc(outputs, labels)
        # acc_list.append(acc)
        # auc_list.append(auc)
        # losses[0].update(loss_dsn.data.item(), outputs.shape[0])
        # acc = accuracy(outputs, labels, topk=(1,))[0]
        # acc = accuracy(outputs_dsn, labels, topk=(1,))[0]
        # top1[0].update(acc.item())
        # print_outputs(outputs, labels, case_id_list)

        confidence_last = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
        for patch_step in range(1, args.T):
            if args.train_stage == 1:
                action = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
            else:
                if patch_step == 1:
                    action = ppo.select_action(states.to(0), memory, restart_batch=True)
                else:
                    action = ppo.select_action(states.to(0), memory)

            feats = get_feats(feat_list, cluster_list, action_sequence=action, feat_size=args.feat_size)
            # if args.mix_up:
            #     feats = mixup(feats, args.alpha)[0]
            outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
            outputs_dsn = dsn_fc(outputs)
            outputs = fc(outputs, restart=False)

            ins_loss = 0
            for r in result_dict:
                ins_loss = ins_loss + r['instance_loss']
            ins_loss = ins_loss / len(feat_list)
            loss = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * ins_loss
            loss_cla.append(loss)
            losses[patch_step].update(loss.data.item(), len(feat_list))

            loss_dsn = args.bag_weight * criterion(outputs_dsn, labels) + (1 - args.bag_weight) * ins_loss
            loss_list_dsn.append(loss_dsn)

            # acc, auc = acc_auc(outputs, labels)
            # acc_list.append(acc)
            # auc_list.append(auc)
            # acc = accuracy(outputs, labels, topk=(1,))[0]
            # top1[patch_step].update(acc.item())

            confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1,
                                                                                                            -1)
            reward = confidence - confidence_last
            confidence_last = confidence

            reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
            memory.rewards.append(reward)
        memory.clear_memory()
        acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)

    return losses[-1].avg, acc, auc, precision, recall, f1_score, outputs, labels, case_id_list


def train_DSMIL(args, epoch, train_set, model, model_prime, fc, ppo, memory, criterion, optimizer, scheduler):
    print(f"train_DSMIL...")
    assert args.batch_size == 1
    length = len(train_set)
    print(f"data length: {length}")
    train_set.shuffle()

    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    if args.train_stage == 2:
        model_prime.eval()
        model.eval()
        fc.eval()
    else:
        if args.train_model_prime:
            model_prime.train()
        else:
            model_prime.eval()
        model.train()
        fc.train()

    dsn_fc_prime = model_prime.module.b_classifier.fcc
    # print(f"dsn_fc_prime: {dsn_fc_prime}")
    dsn_fc = model.module.b_classifier.fcc
    # print(f"dsn_fc: {dsn_fc}")

    progress_bar = tqdm(range(args.num_data))
    feat_list, cluster_list, label_list, step = [], [], [], 0
    batch_idx = 0
    labels_list, outputs_list = [], []
    for data_idx in progress_bar:
        loss_cla = []
        loss_list_dsn = []

        feat, cluster, label, case_id = train_set[data_idx % length]
        assert len(feat.shape) == 2, f"{feat.shape}"
        feat = feat.unsqueeze(0).to(args.device)
        label = label.unsqueeze(0).to(args.device)

        feat_list.append(feat)
        cluster_list.append(cluster)
        label_list.append(label)

        step += 1
        if step == args.batch_size or data_idx == args.num_data - 1:
            labels = torch.cat(label_list)
            action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
            feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
            # print(f"feats.shape: {feats.shape}")
            if args.mix_up:
                feats = mixup(feats, args.alpha)[0]
            if args.train_model_prime and args.train_stage != 2:
                outputs_ins, outputs, states = model_prime(feats)
                states = torch.mean(states, dim=1)
                # print(outputs_ins.shape)
                outputs_max, _ = torch.max(outputs_ins, 0, keepdim=True)
                # print(f"outputs_max: {outputs_max.shape}")
                outputs_dsn = dsn_fc_prime(outputs).view(1, -1)
                outputs = torch.mean(outputs, dim=1)
                outputs = fc(outputs, restart=True)
            else:
                with torch.no_grad():
                    outputs_ins, outputs, states = model_prime(feats)
                    states = torch.mean(states, dim=1)
                    outputs_max, _ = torch.max(outputs_ins, 0, keepdim=True)
                    outputs_dsn = dsn_fc_prime(outputs).view(1, -1)
                    outputs = torch.mean(outputs, dim=1)
                    outputs = fc(outputs, restart=True)

            loss_max = criterion(outputs_max, labels)

            loss_prime = 0.5 * criterion(outputs, labels) + 0.5 * loss_max
            loss_cla.append(loss_prime)

            loss_dsn = 0.5 * criterion(outputs_dsn, labels) + 0.5 * loss_max
            loss_list_dsn.append(loss_dsn)

            losses[0].update(loss_prime.data.item(), outputs.shape[0])
            acc = accuracy(outputs, labels, topk=(1,))[0]
            top1[0].update(acc.item(), outputs.shape[0])

            # RL
            confidence_last = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
            for patch_step in range(1, args.T):
                if args.train_stage == 1:
                    action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
                else:
                    if patch_step == 1:
                        action_sequence = ppo.select_action(states.to(0), memory, restart_batch=True)
                    else:
                        action_sequence = ppo.select_action(states.to(0), memory)

                feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
                if args.mix_up:
                    feats = mixup(feats, args.alpha)[0]

                if args.train_stage != 2:
                    outputs_ins, outputs, states = model(feats)
                    states = torch.mean(states, dim=1)
                    outputs_max, _ = torch.max(outputs_ins, 0, keepdim=True)
                    outputs_dsn = dsn_fc(outputs).view(1, -1)
                    outputs = torch.mean(outputs, dim=1)
                    outputs = fc(outputs, restart=False)
                else:
                    with torch.no_grad():
                        outputs_ins, outputs, states = model(feats)
                        states = torch.mean(states, dim=1)
                        outputs_max, _ = torch.max(outputs_ins, 0, keepdim=True)
                        outputs_dsn = dsn_fc(outputs).view(1, -1)
                        outputs = torch.mean(outputs, dim=1)
                        outputs = fc(outputs, restart=False)
                loss_max = criterion(outputs_max, labels)

                loss = 0.5 * criterion(outputs, labels) + 0.5 * loss_max
                loss_cla.append(loss)
                losses[patch_step].update(loss.data.item(), len(feat_list))

                loss_dsn = 0.5 * criterion(outputs_dsn, labels) + 0.5 * loss_max
                loss_list_dsn.append(loss_dsn)

                acc = accuracy(outputs, labels, topk=(1,))[0]
                top1[patch_step].update(acc.item(), outputs.shape[0])

                confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1,
                                                                                                                -1)
                reward = confidence - confidence_last
                confidence_last = confidence

                reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                memory.rewards.append(reward)

            # print(f"loss_cla:\n{loss_cla}")
            # print(f"loss_list_dsn:\n{loss_list_dsn}")
            loss = (sum(loss_cla) + args.dsn_ratio * sum(loss_list_dsn)) / args.T
            if args.train_stage != 2:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                ppo.update(memory)
            memory.clear_memory()
            torch.cuda.empty_cache()

            labels_list.append(labels.detach())
            outputs_list.append(outputs.detach())

            feat_list, cluster_list, label_list, step = [], [], [], 0
            batch_idx += 1
            progress_bar.set_description(
                f"Train Epoch: {epoch + 1:2}/{args.epochs:2}. Iter: {batch_idx:3}/{args.eval_step:3}. "
                f"Loss: {losses[-1].avg:.4f}. Acc: {top1[-1].avg:.4f}"
            )
            progress_bar.update()

            # _acc = [acc.avg for acc in top1]
            # print('accuracy of each step:')
            # print(_acc)

            # _reward = [reward.avg for reward in reward_list]
            # print('reward of each step:')
            # print(_reward)

    progress_bar.close()
    if args.train_stage != 2 and scheduler is not None and epoch >= args.warmup:
        scheduler.step()

    labels = torch.cat(labels_list)
    outputs = torch.cat(outputs_list)
    acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)
    return losses[-1].avg, acc, auc, precision, recall, f1_score


def test_DSMIL(args, test_set, model, model_prime, fc, ppo, memory, criterion, mode):
    losses = [AverageMeter() for _ in range(args.T)]
    # top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]
    model_prime.eval()
    model.eval()
    fc.eval()
    dsn_fc_prime = model_prime.module.b_classifier.fcc
    dsn_fc = model.module.b_classifier.fcc
    acc_list, auc_list = [], []
    with torch.no_grad():

        # progress_bar = tqdm(test_set)
        feat_list, cluster_list, label_list, case_id_list, step = [], [], [], [], 0
        # batch_idx = 0
        # labels_list, outputs_list = [], []
        for data_idx, (feat, cluster, label, case_id) in enumerate(test_set):
            loss_cla = []
            loss_list_dsn = []

            feat = feat.unsqueeze(0).to(args.device)
            label = label.unsqueeze(0).to(args.device)
            feat_list.append(feat)
            cluster_list.append(cluster)
            label_list.append(label)
            case_id_list.append(case_id)

        labels = torch.cat(label_list)
        action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
        feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
        # if args.mix_up:
        #     feats = mixup(feats, args.alpha)[0]

        # print(f"feats.shape: {feats.shape}")
        outputs_ins, outputs, states = model_prime(feats)
        states = torch.mean(states, dim=1)
        # print(f"outputs_ins.shape: {outputs_ins.shape}")
        # print(f"outputs.shape: {outputs.shape}")
        # print(f"states.shape: {states.shape}")
        outputs_max = []
        for ins in outputs_ins:
            m_ins, _ = torch.max(ins, 0, keepdim=True)
            outputs_max.append(m_ins)
        outputs_max = torch.cat(outputs_max)
        # print(f"outputs_max.shape: {outputs_max.shape}")
        # outputs_max, _ = torch.max(outputs_ins, 0, keepdim=True)
        outputs_dsn = dsn_fc_prime(outputs).view(len(feat_list), -1)
        outputs = torch.mean(outputs, dim=1)
        outputs = fc(outputs, restart=True)

        loss_max = criterion(outputs_max, labels)

        loss_prime = 0.5 * criterion(outputs, labels) + 0.5 * loss_max
        loss_cla.append(loss_prime)

        loss_dsn = 0.5 * criterion(outputs_dsn, labels) + 0.5 * loss_max
        loss_list_dsn.append(loss_dsn)

        losses[0].update(loss_prime.data.item(), outputs.shape[0])
        # acc, auc = acc_auc(outputs, labels)
        # acc_list.append(acc)
        # auc_list.append(auc)
        # losses[0].update(loss_dsn.data.item(), outputs.shape[0])
        # acc = accuracy(outputs, labels, topk=(1,))[0]
        # acc = accuracy(outputs_dsn, labels, topk=(1,))[0]
        # top1[0].update(acc.item())
        # print_outputs(outputs, labels, case_id_list)

        confidence_last = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
        for patch_step in range(1, args.T):
            if args.train_stage == 1:
                action = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
            else:
                if patch_step == 1:
                    action = ppo.select_action(states.to(0), memory, restart_batch=True)
                else:
                    action = ppo.select_action(states.to(0), memory)

            feats = get_feats(feat_list, cluster_list, action_sequence=action, feat_size=args.feat_size)
            # if args.mix_up:
            #     feats = mixup(feats, args.alpha)[0]
            outputs_ins, outputs, states = model(feats)
            states = torch.mean(states, dim=1)
            outputs_max = []
            for ins in outputs_ins:
                m_ins, _ = torch.max(ins, 0, keepdim=True)
                outputs_max.append(m_ins)
            outputs_max = torch.cat(outputs_max)
            # outputs_max, _ = torch.max(outputs_ins, 0, keepdim=True)
            outputs_dsn = dsn_fc(outputs).view(len(feat_list), -1)
            outputs = torch.mean(outputs, dim=1)
            outputs = fc(outputs, restart=False)

            loss_max = criterion(outputs_max, labels)

            loss = 0.5 * criterion(outputs, labels) + 0.5 * loss_max
            loss_cla.append(loss)
            losses[patch_step].update(loss.data.item(), len(feat_list))

            loss_dsn = 0.5 * criterion(outputs_dsn, labels) + 0.5 * loss_max
            loss_list_dsn.append(loss_dsn)

            # acc, auc = acc_auc(outputs, labels)
            # acc_list.append(acc)
            # auc_list.append(auc)
            # acc = accuracy(outputs, labels, topk=(1,))[0]
            # top1[patch_step].update(acc.item())

            confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1,
                                                                                                            -1)
            reward = confidence - confidence_last
            confidence_last = confidence

            reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
            memory.rewards.append(reward)
        memory.clear_memory()
        acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)
    return losses[-1].avg, acc, auc, precision, recall, f1_score, outputs, labels, case_id_list


def train_ABMIL(args, epoch, train_set, model, model_prime, fc, ppo, memory, criterion, optimizer, scheduler):
    print(f"train_ABMIL...")
    length = len(train_set)
    print(f"data length: {length}")
    train_set.shuffle()

    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    if args.train_stage == 2:
        model_prime.eval()
        model.eval()
        fc.eval()
    else:
        if args.train_model_prime:
            model_prime.train()
        else:
            model_prime.eval()
        model.train()
        fc.train()

    dsn_fc_prime = model_prime.module.fc
    # print(f"dsn_fc_prime: {dsn_fc_prime}")
    dsn_fc = model.module.fc
    # print(f"dsn_fc: {dsn_fc}")

    progress_bar = tqdm(range(args.num_data))
    feat_list, cluster_list, label_list, step = [], [], [], 0
    batch_idx = 0
    labels_list, outputs_list = [], []
    for data_idx in progress_bar:
        loss_cla = []
        loss_list_dsn = []

        feat, cluster, label, case_id = train_set[data_idx % length]
        assert len(feat.shape) == 2, f"{feat.shape}"
        feat = feat.unsqueeze(0).to(args.device)
        label = label.unsqueeze(0).to(args.device)

        feat_list.append(feat)
        cluster_list.append(cluster)
        label_list.append(label)

        step += 1
        if step == args.batch_size or data_idx == args.num_data - 1:
            labels = torch.cat(label_list)
            action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
            feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
            if args.mix_up:
                feats = mixup(feats, args.alpha)[0]
            if args.train_model_prime and args.train_stage != 2:
                outputs, states = model_prime(feats)
                outputs_dsn = dsn_fc_prime(outputs)
                outputs = fc(outputs, restart=True)
            else:
                with torch.no_grad():
                    outputs, states = model_prime(feats)
                    outputs_dsn = dsn_fc_prime(outputs)
                    outputs = fc(outputs, restart=True)

            loss_prime = criterion(outputs, labels)
            # print(f"loss_prime: {loss_prime}")
            loss_cla.append(loss_prime)

            loss_dsn = criterion(outputs_dsn, labels)
            loss_list_dsn.append(loss_dsn)

            losses[0].update(loss_prime.data.item(), outputs.shape[0])
            acc = accuracy(outputs, labels, topk=(1,))[0]
            top1[0].update(acc.item())

            # RL
            confidence_last = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
            for patch_step in range(1, args.T):
                if args.train_stage == 1:
                    action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
                else:
                    if patch_step == 1:
                        action_sequence = ppo.select_action(states.to(0), memory, restart_batch=True)
                    else:
                        action_sequence = ppo.select_action(states.to(0), memory)

                feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
                if args.mix_up:
                    feats = mixup(feats, args.alpha)[0]

                if args.train_stage != 2:
                    outputs, states = model(feats)
                    outputs_dsn = dsn_fc(outputs)
                    outputs = fc(outputs, restart=False)
                else:
                    with torch.no_grad():
                        outputs, states = model(feats)
                        outputs_dsn = dsn_fc(outputs)
                        outputs = fc(outputs, restart=False)
                loss = criterion(outputs, labels)
                loss_cla.append(loss)
                losses[patch_step].update(loss.data.item(), len(feat_list))

                loss_dsn = criterion(outputs_dsn, labels)
                loss_list_dsn.append(loss_dsn)

                acc = accuracy(outputs, labels, topk=(1,))[0]
                top1[patch_step].update(acc.item())

                confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1,
                                                                                                                -1)
                reward = confidence - confidence_last
                confidence_last = confidence

                reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                memory.rewards.append(reward)

            # print(f"loss_cla:\n{loss_cla}")
            # print(f"loss_list_dsn:\n{loss_list_dsn}")
            loss = (sum(loss_cla) + args.dsn_ratio * sum(loss_list_dsn)) / args.T
            if args.train_stage != 2:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                ppo.update(memory)
            memory.clear_memory()
            torch.cuda.empty_cache()

            labels_list.append(labels.detach())
            outputs_list.append(outputs.detach())

            feat_list, cluster_list, label_list, step = [], [], [], 0
            batch_idx += 1
            progress_bar.set_description(
                f"Train Epoch: {epoch + 1:2}/{args.epochs:2}. Iter: {batch_idx:3}/{args.eval_step:3}. "
                f"Loss: {losses[-1].avg:.4f}. Acc: {top1[-1].avg:.4f}"
            )
            progress_bar.update()

            # _acc = [acc.avg for acc in top1]
            # print('accuracy of each step:')
            # print(_acc)

            # _reward = [reward.avg for reward in reward_list]
            # print('reward of each step:')
            # print(_reward)

    progress_bar.close()
    if args.train_stage != 2 and scheduler is not None and epoch >= args.warmup:
        scheduler.step()

    labels = torch.cat(labels_list)
    outputs = torch.cat(outputs_list)
    acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)
    # predict = torch.softmax(outputs, dim=1)
    # auc = roc_auc_score(labels.cpu().numpy(), predict[:, 1].cpu().numpy())

    return losses[-1].avg, acc, auc, precision, recall, f1_score


def test_ABMIL(args, test_set, model, model_prime, fc, ppo, memory, criterion, mode):
    losses = [AverageMeter() for _ in range(args.T)]
    # top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]
    model_prime.eval()
    model.eval()
    fc.eval()
    dsn_fc_prime = model_prime.module.fc
    dsn_fc = model.module.fc
    acc_list, auc_list = [], []
    with torch.no_grad():

        # progress_bar = tqdm(test_set)
        feat_list, cluster_list, label_list, case_id_list, step = [], [], [], [], 0
        # batch_idx = 0
        # labels_list, outputs_list = [], []
        for data_idx, (feat, cluster, label, case_id) in enumerate(test_set):
            loss_cla = []
            loss_list_dsn = []

            feat = feat.unsqueeze(0).to(args.device)
            label = label.unsqueeze(0).to(args.device)
            feat_list.append(feat)
            cluster_list.append(cluster)
            label_list.append(label)
            case_id_list.append(case_id)

        labels = torch.cat(label_list)
        action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
        feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
        # if args.mix_up:
        #     feats = mixup(feats, args.alpha)[0]

        outputs, states = model_prime(feats)
        outputs_dsn = dsn_fc_prime(outputs)
        outputs = fc(outputs, restart=True)

        loss_prime = criterion(outputs, labels)
        loss_cla.append(loss_prime)

        loss_dsn = criterion(outputs_dsn, labels)
        loss_list_dsn.append(loss_dsn)

        losses[0].update(loss_prime.data.item(), outputs.shape[0])
        # acc, auc = acc_auc(outputs, labels)
        # acc_list.append(acc)
        # auc_list.append(auc)
        # losses[0].update(loss_dsn.data.item(), outputs.shape[0])
        # acc = accuracy(outputs, labels, topk=(1,))[0]
        # acc = accuracy(outputs_dsn, labels, topk=(1,))[0]
        # top1[0].update(acc.item())
        # print_outputs(outputs, labels, case_id_list)

        confidence_last = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
        for patch_step in range(1, args.T):
            if args.train_stage == 1:
                action = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
            else:
                if patch_step == 1:
                    action = ppo.select_action(states.to(0), memory, restart_batch=True)
                else:
                    action = ppo.select_action(states.to(0), memory)

            feats = get_feats(feat_list, cluster_list, action_sequence=action, feat_size=args.feat_size)
            # if args.mix_up:
            #     feats = mixup(feats, args.alpha)[0]
            outputs, states = model(feats)
            outputs_dsn = dsn_fc(outputs)
            outputs = fc(outputs, restart=False)

            loss = criterion(outputs, labels)
            loss_cla.append(loss)
            losses[patch_step].update(loss.data.item(), len(feat_list))

            loss_dsn = criterion(outputs_dsn, labels)
            loss_list_dsn.append(loss_dsn)

            # acc, auc = acc_auc(outputs, labels)
            # acc_list.append(acc)
            # auc_list.append(auc)
            # acc = accuracy(outputs, labels, topk=(1,))[0]
            # top1[patch_step].update(acc.item())

            confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1,
                                                                                                            -1)
            reward = confidence - confidence_last
            confidence_last = confidence

            reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
            memory.rewards.append(reward)
        memory.clear_memory()
        acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)

    return losses[-1].avg, acc, auc, precision, recall, f1_score, outputs, labels, case_id_list


# Basic Functions ------------------------------------------------------------------------------------------------------
def train(args, train_set, valid_set, test_set, model, model_prime, fc, ppo, memory, criterion, optimizer, scheduler,
          tb_writer, save_dir):
    # Init variables
    best_train_acc = BestVariable(order='max')
    best_valid_acc = BestVariable(order='max')
    best_test_acc = BestVariable(order='max')
    best_train_auc = BestVariable(order='max')
    best_valid_auc = BestVariable(order='max')
    best_test_auc = BestVariable(order='max')
    best_train_loss = BestVariable(order='min')
    best_valid_loss = BestVariable(order='min')
    best_test_loss = BestVariable(order='min')
    best_score = BestVariable(order='max')
    final_loss, final_acc, final_auc, final_precision, final_recall, final_f1_score, final_epoch = 0., 0., 0., 0., 0., 0., 0
    header = ['epoch', 'train', 'valid', 'test', 'best_train', 'best_valid', 'best_test']
    losses_csv = CSVWriter(filename=Path(save_dir) / 'losses.csv', header=header)
    accs_csv = CSVWriter(filename=Path(save_dir) / 'accs.csv', header=header)
    aucs_csv = CSVWriter(filename=Path(save_dir) / 'aucs.csv', header=header)
    results_csv = CSVWriter(filename=Path(save_dir) / 'results.csv',
                            header=['epoch', 'final_epoch', 'final_loss', 'final_acc', 'final_auc', 'final_precision', 'final_recall', 'final_f1_score'])

    best_model = copy.deepcopy({'state_dict': model.state_dict()})
    early_stop = EarlyStop(max_num_accordance=args.patience) if args.patience is not None else None

    for epoch in range(args.epochs):
        print(f"Training Stage: {args.train_stage}, lr:")
        if optimizer is not None:
            for k, group in enumerate(optimizer.param_groups):
                print(f"group[{k}]: {group['lr']}")
        train_loss, train_acc, train_auc, train_precision, train_recall, train_f1_score = \
            TRAIN[args.arch](args, epoch, train_set, model, model_prime, fc, ppo, memory, criterion, optimizer,
                             scheduler)
        valid_loss, valid_acc, valid_auc, valid_precision, valid_recall, valid_f1_score, *_ = \
            TEST[args.arch](args, valid_set, model, model_prime, fc, ppo, memory, criterion, mode='Valid')
        test_loss, test_acc, test_auc, test_precision, test_recall, test_f1_score, *_ = \
            TEST[args.arch](args, test_set, model, model_prime, fc, ppo, memory, criterion, mode='Test ')
        if args.nni:
            score = (test_acc + test_auc) / 2
            nni.report_intermediate_result(score)

        # Write to tensorboard
        if tb_writer is not None:
            tb_writer.add_scalar('train/1.train_loss', train_loss, epoch)
            tb_writer.add_scalar('test/2.test_loss', valid_loss, epoch)

        # Choose the best result
        if args.picked_method == 'acc':
            is_best = best_valid_acc.compare(valid_acc)
        elif args.picked_method == 'loss':
            is_best = best_valid_loss.compare(valid_loss)
        elif args.picked_method == 'auc':
            is_best = best_valid_auc.compare(valid_auc)
        elif args.picked_method == 'score':
            score = get_score(valid_acc, valid_auc, valid_precision, valid_recall, valid_f1_score)
            is_best = best_score.compare(score, epoch + 1, inplace=True)
        else:
            raise ValueError(f"picked_method error. ")
        if is_best:
            final_epoch = epoch + 1
            final_loss = test_loss
            final_acc = test_acc
            final_auc = test_auc
            final_precision = test_precision
            final_recall = test_recall
            final_f1_score = test_f1_score
        # Compute best result
        best_train_acc.compare(train_acc, epoch + 1, inplace=True)
        best_valid_acc.compare(valid_acc, epoch + 1, inplace=True)
        best_test_acc.compare(test_acc, epoch + 1, inplace=True)
        best_train_loss.compare(train_loss, epoch + 1, inplace=True)
        best_valid_loss.compare(valid_loss, epoch + 1, inplace=True)
        best_test_loss.compare(test_loss, epoch + 1, inplace=True)
        best_train_auc.compare(train_auc, epoch + 1, inplace=True)
        best_valid_auc.compare(valid_auc, epoch + 1, inplace=True)
        best_test_auc.compare(test_auc, epoch + 1, inplace=True)

        state = {
            'epoch': epoch + 1,
            'model_state_dict': model.module.state_dict(),
            'model_prime_state_dict': model_prime.module.state_dict(),
            'fc': fc.state_dict(),
            'acc': valid_acc,
            'best_acc': best_valid_acc.best,
            'optimizer': optimizer.state_dict() if optimizer else None,
            'ppo_optimizer': ppo.optimizer.state_dict() if ppo else None,
            'policy': ppo.policy.state_dict() if ppo else None,
        }
        if is_best:
            best_model = copy.deepcopy(state)
            if args.save_model:
                save_checkpoint(state, is_best, str(save_dir))

        # Save
        losses_csv.write_row([epoch + 1, train_loss, valid_loss, test_loss,
                              (best_train_loss.best, best_train_loss.epoch),
                              (best_valid_loss.best, best_valid_loss.epoch),
                              (best_test_loss.best, best_test_loss.epoch)])
        accs_csv.write_row([epoch + 1, train_acc, valid_acc, test_acc,
                            (best_train_acc.best, best_train_acc.epoch),
                            (best_valid_acc.best, best_valid_acc.epoch),
                            (best_test_acc.best, best_test_acc.epoch)])
        aucs_csv.write_row([epoch + 1, train_auc, valid_auc, test_auc,
                            (best_train_auc.best, best_train_auc.epoch),
                            (best_valid_auc.best, best_valid_auc.epoch),
                            (best_test_auc.best, best_test_auc.epoch)])
        results_csv.write_row([epoch + 1, final_epoch, test_loss, test_acc, test_auc, test_precision, test_recall, test_f1_score])

        print(
            f"Train acc: {train_acc:.4f}, Best: {best_train_acc.best:.4f}, Epoch: {best_train_acc.epoch:2}, "
            f"AUC: {train_auc:.4f}, Best: {best_train_auc.best:.4f}, Epoch: {best_train_auc.epoch:2}, "
            f"Loss: {train_loss:.4f}, Best: {best_train_loss.best:.4f}, Epoch: {best_train_loss.epoch:2}\n"
            f"Valid acc: {valid_acc:.4f}, Best: {best_valid_acc.best:.4f}, Epoch: {best_valid_acc.epoch:2}, "
            f"AUC: {valid_auc:.4f}, Best: {best_valid_auc.best:.4f}, Epoch: {best_valid_auc.epoch:2}, "
            f"Loss: {valid_loss:.4f}, Best: {best_valid_loss.best:.4f}, Epoch: {best_valid_loss.epoch:2}\n"
            f"Test  acc: {test_acc:.4f}, Best: {best_test_acc.best:.4f}, Epoch: {best_test_acc.epoch:2}, "
            f"AUC: {test_auc:.4f}, Best: {best_test_auc.best:.4f}, Epoch: {best_test_auc.epoch:2}, "
            f"Loss: {test_loss:.4f}, Best: {best_test_loss.best:.4f}, Epoch: {best_test_loss.epoch:2}\n"
            f"Final Epoch: {final_epoch:2}, Final acc: {final_acc:.4f}, Final AUC: {final_auc:.4f}, Final Loss: {final_loss:.4f}\n"
        )

        # Early Stop
        if early_stop is not None and (train_acc > args.wait_thresh or epoch > args.wait_epoch):
            early_stop.update((best_valid_acc.best, best_valid_loss.best))
            if early_stop.is_stop():
                break

    if tb_writer is not None:
        tb_writer.close()
    # if args.nni:
    #     score = ((1 - test_loss) + test_acc + test_auc) / 3
    #     nni.report_final_result(score)
    return best_model


def test(args, test_set, model, model_prime, fc, ppo, memory, criterion, mode='Test '):
    model.eval()
    with torch.no_grad():
        loss, acc, auc, precision, recall, f1_score, outputs_tensor, labels_tensor, case_id_list = \
            TEST[args.arch](args, test_set, model, model_prime, fc, ppo, memory, criterion, mode)
        prob = torch.softmax(outputs_tensor, dim=1)
        _, pred = torch.max(prob, dim=1)
        preds = pd.DataFrame(columns=['label', 'pred', 'correct', *[f'prob{i}' for i in range(prob.shape[1])]])
        for i in range(len(case_id_list)):
            preds.loc[case_id_list[i]] = [
                labels_tensor[i].item(),
                pred[i].item(),
                labels_tensor[i].item() == pred[i].item(),
                *[prob[i][j].item() for j in range(prob.shape[1])],
            ]
        preds.index.rename('case_id', inplace=True)

    return loss, acc, auc, precision, recall, f1_score, preds


def run(args):
    # Configures
    init_seeds(args.seed)

    # 如果文件夹存在且不覆盖, 则顺序生成新的文件夹一'_'分隔
    if args.save_dir is None:
        create_save_dir(args)
    else:
        args.save_dir = str(Path(args.base_save_dir) / args.save_dir)
    args.save_dir = increment_path(Path(args.save_dir), exist_ok=args.exist_ok, sep='_')  # increment run
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    # print(args.save_dir)

    # Save arguments
    with open(Path(args.save_dir) / 'args.yaml', 'w') as fp:
        yaml.dump(vars(args), fp, sort_keys=False)
    print(args, '\n')

    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    # Dataset, Model, Criterion, Optimizer and Scheduler
    datasets, dim_patch, train_length = get_datasets(args)
    args.num_data = train_length * args.data_repeat
    args.eval_step = int(args.num_data / args.batch_size)
    print(f"train_length: {train_length}, epoch_step: {args.num_data}, eval_step: {args.eval_step}")
    model, model_prime, fc, ppo = create_model(args, dim_patch)
    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model, model_prime, fc)
    scheduler = get_scheduler(args, optimizer)

    # TensorBoard
    tb_writer = SummaryWriter(args.save_dir) if args.use_tensorboard else None

    # Start training
    memory = rlmil.Memory()
    best_model = train(args, datasets['train'], datasets['valid'], datasets['test'], model, model_prime, fc, ppo,
                       memory, criterion, optimizer, scheduler, tb_writer, args.save_dir)
    model.module.load_state_dict(best_model['model_state_dict'])
    model_prime.module.load_state_dict(best_model['model_prime_state_dict'])
    fc.load_state_dict(best_model['fc'])
    if ppo is not None:
        ppo.policy.load_state_dict(best_model['policy'])
    loss, acc, auc, precision, recall, f1_score, preds = \
        test(args, datasets['test'], model, model_prime, fc, ppo, memory, criterion, mode='Pred')

    if args.nni:
        score = (acc + auc) / 2
        nni.report_final_result(score)

    # Save results
    # pred.to_csv(str(Path(args.save_dir) / 'pred.csv'))
    # print(f'Pred | Loss: {loss:.4f}, Acc: {acc:.4f}, AUC: {auc}\nPredicted Ending.\n')
    preds.to_csv(str(Path(args.save_dir) / 'pred.csv'))
    final_res = pd.DataFrame(columns=['loss', 'acc', 'auc', 'precision', 'recall', 'f1_score'])
    final_res.loc[f'seed{args.seed}'] = [loss, acc, auc, precision, recall, f1_score]
    final_res.to_csv(str(Path(args.save_dir) / 'final_res.csv'))
    print(f'{final_res}\nPredicted Ending.\n')


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='CAMELYON16_20x_s256_rgb_simclr_resnet18_v0_cluster10',
                        help='Specify the dataset used')
    parser.add_argument('--data_csv', type=str,
                        default='/data4/wwu/CAMELYON16/patch/patch_20x_s256_rgb/features_wo_norm/simclr_resnet18_v0_cluster_10.csv',
                        help='')
    parser.add_argument('--data_split_json', type=str, default='/data4/wwu/CAMELYON16/data_split_985.json',
                        help="当数据划分方式(--data_split)为fixed时，由该变量传递json格式的文件路径")
    parser.add_argument('--train_data', type=str, default='train')
    parser.add_argument('--preload', action='store_true', default=False,
                        help="决定是否预加载数据")
    parser.add_argument('--data_repeat', type=int, default=1)
    parser.add_argument('--feat_size', default=1024, type=int,
                        help='size of local map (we recommend 96 / 128 / 144)')
    parser.add_argument('--mix_up', action='store_true', default=False)
    parser.add_argument('--alpha', type=float, default=0.8)
    # Train
    parser.add_argument('--train_method', type=str, default='scratch', choices=['scratch', 'finetune', 'linear'])
    parser.add_argument('--train_stage', default=1, type=int,
                        help="select training stage, see our paper for details \
                              stage-1 : warm-up \
                              stage-2 : learn to select patches with RL \
                              stage-3 : finetune CNNs")
    parser.add_argument('--T', default=6, type=int,
                        help='maximum length of the sequence of Glance + Focus')
    parser.add_argument('--checkpoint_path', default=None, type=str,
                        help='path to the stage-2/3 checkpoint (for training stage-2/3)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='path to the stage-1 checkpoint (for finetune stage-1)')
    parser.add_argument('--train_model_prime', action='store_true', default=True)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['AdamW', 'Adam', 'SGD', 'RMSprop'],
                        help="指定训练时使用的optimizer")
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'StepLR', 'CosineAnnealingLR'],
                        help="指定训练时使用的scheduler, 默认为None不使用")
    parser.add_argument('--batch_size', type=int, default=1,
                        help='指定训练batch_size个数据时进行一次参数更新')
    parser.add_argument('--epochs', type=int, default=40,
                        help="指定最多训练多少个epochs")
    parser.add_argument('--backbone_lr', default=1e-4, type=float)
    parser.add_argument('--fc_lr', default=1e-4, type=float)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="SGD优化器中的参数momentum, 默认0.5")
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--beta1', type=float, default=0.9,
                        help="Adam优化器中的参数momentum")
    parser.add_argument('--beta2', type=float, default=0.999,
                        help="Adam优化器中的参数momentum")
    parser.add_argument('--warmup', default=0, type=float,
                        help='当使用lr_scheduler时, 预热warmup个epochs后才更新lr')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='所有优化器的weight decay')
    parser.add_argument('--picked_method', type=str, default='score',
                        help="指定挑选最优模型的方式, 默认为acc, 也可以是loss")
    parser.add_argument('--patience', type=int, default=None,
                        help="停止训练的容忍度, 默认为10, 即验证集的best result在10个epochs内都没有变化则停止训练")

    # Architecture
    parser.add_argument('--arch', default='ABMIL', type=str, choices=MODELS, help='model name')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--dsn_ratio', type=float, default=1.)
    # Architecture - PPO
    parser.add_argument('--policy_hidden_dim', type=int, default=256)
    parser.add_argument('--policy_conv', action='store_true', default=False)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=0.00001)
    parser.add_argument('--ppo_gamma', type=float, default=0.3)
    parser.add_argument('--K_epochs', type=int, default=2)
    # Architecture - Full_layer
    parser.add_argument('--feature_num', type=int, default=512)
    parser.add_argument('--fc_hidden_dim', type=int, default=512)
    parser.add_argument('--fc_rnn', action='store_true', default=True)
    parser.add_argument('--load_fc', action='store_true', default=False)
    # Architecture - ABMIL
    parser.add_argument('--L', type=int, default=512)
    parser.add_argument('--D', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.0)
    # CLAM
    parser.add_argument('--size_arg', type=str, default='small', choices=['small', 'big'])
    parser.add_argument('--k_sample', type=int, default=8)
    parser.add_argument('--bag_weight', type=float, default=0.7)
    # Loss
    parser.add_argument('--loss', default='CrossEntropyLoss', type=str, choices=LOSSES,
                        help='loss name')
    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help="是否使用TensorBoard")
    # Save
    parser.add_argument('--base_save_dir', type=str, default='/data11/wwu/RLMIL/results')
    parser.add_argument('--save_dir', type=str, default=None, help="保存实验结果的路径")
    parser.add_argument('--save_dir_flag', type=str, default=None)
    parser.add_argument('--exist_ok', action='store_true', default=False,
                        help="是否覆盖 --save_dir 内的内容")
    parser.add_argument('--save_model', action='store_true', default=False, help="是否需要保存模型")
    # Global
    parser.add_argument('--device', default='2',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=985,
                        help="全局的随机种子")
    parser.add_argument('--seek', type=str, default=None)
    parser.add_argument('--seek_thresh', type=float, default=0.55)
    parser.add_argument('--nni', action='store_true', default=False)
    args = parser.parse_args()

    if args.nni:
        try:
            tuner_params = nni.get_next_parameter()
            args = merge_parameter(args, tuner_params)
        except Exception as exception:
            raise exception

    run(args)


if __name__ == '__main__':
    # Pandas print setting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    torch.set_num_threads(1)

    # Global variables
    MODELS = ['ABMIL', 'MaxPooling', 'ABMIL_S', 'DSMIL', 'CLAM_SB']

    LOSSES = ['CrossEntropyLoss']

    TRAIN = {
        'ABMIL': train_ABMIL,
        'ABMIL_S': train_ABMIL,
        'MaxPooling': train_ABMIL,
        'DSMIL': train_DSMIL,
        'CLAM_SB': train_CLAM,
    }
    TEST = {
        'ABMIL': test_ABMIL,
        'ABMIL_S': test_ABMIL,
        'MaxPooling': test_ABMIL,
        'DSMIL': test_DSMIL,
        'CLAM_SB': test_CLAM,
    }

    main()
