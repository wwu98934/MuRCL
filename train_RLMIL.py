import os
import copy
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import WSIWithCluster, get_feats
from utils.general import AverageMeter, CSVWriter, EarlyStop, increment_path, BestVariable, accuracy, init_seeds, \
    load_json, get_metrics, get_score, save_checkpoint
from models import rlmil, abmil, clam, dsmil


def create_save_dir(args):
    dir1 = f'{args.dataset}_np_{args.feat_size}'
    dir2 = f'RLMIL'
    rlmil_setting = [
        f'T{args.T}',
        f'as{args.action_std}',
        f'pg{args.ppo_gamma}',
        f'phd{args.policy_hidden_dim}',
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
    elif args.arch in ['CLAM_SB']:
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
        model = abmil.ABMIL(
            dim_in=dim_patch,
            L=args.L,
            D=args.D,
            dim_out=args.num_classes,
            dropout=args.dropout,
        )
        args.feature_num = args.L
    elif args.arch == 'DSMIL':
        model = dsmil.build_dsmil(dim_feat=dim_patch, num_classes=args.num_classes)
        args.feature_num = dim_patch
    elif args.arch == 'CLAM_SB':
        model = clam.CLAM_SB(
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
            assert args.checkpoint_pretrained is not None and Path(
                args.checkpoint).exists(), f"{args.checkpoint} is not exists!"

            checkpoint = torch.load(args.checkpoint)
            model_state_dict = checkpoint['model_state_dict']
            # for k in list(model_state_dict.keys()):
            #     print(f"key: {k}")
            for k in list(model_state_dict.keys()):
                if k.startswith('encoder') and not k.startswith('encoder.fc') and not k.startswith(
                        'encoder.classifiers'):
                    model_state_dict[k[len('encoder.'):]] = model_state_dict[k]
                del model_state_dict[k]
            # for k in list(model_state_dict.keys()):
            #     print(f"key: {k}")
            msg_model = model.load_state_dict(model_state_dict, strict=False)
            print(f"msg_model missing_keys: {msg_model.missing_keys}")

            # fix the parameters of  mil encoder
            if args.train_method == 'linear':
                for n, p in model.named_parameters():
                    # print(f"key: {n}")
                    if n.startswith('fc') or n.startswith('classifiers') or n.startswith('instance_classifiers'):
                        print(f"not_fixed_key: {n}")
                    else:
                        p.requires_grad = False

        elif args.train_stage == 2:
            if args.checkpoint_stage is None:
                args.checkpoint_stage = str(Path(args.save_dir).parent / 'stage_1' / 'model_best.pth.tar')
            assert Path(args.checkpoint_stage).exists(), f"{args.checkpoint_stage} is not exist!"

            checkpoint = torch.load(args.checkpoint_stage)
            model.load_state_dict(checkpoint['model_state_dict'])
            fc.load_state_dict(checkpoint['fc'])

            assert args.checkpoint_pretrained is not None and Path(
                args.checkpoint_pretrained).exists(), f"{args.checkpoint_pretrained} is not exists!"
            checkpoint = torch.load(args.checkpoint_pretrained)
            state_dim = args.model_dim
            ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                            action_std=args.action_std,
                            lr=args.ppo_lr,
                            gamma=args.ppo_gamma,
                            K_epochs=args.K_epochs,
                            action_size=args.num_clusters)
            ppo.policy.load_state_dict(checkpoint['policy'])
            ppo.policy_old.load_state_dict(checkpoint['policy'])

        elif args.train_stage == 3:
            if args.checkpoint_stage is None:
                args.checkpoint_stage = str(Path(args.save_dir).parent / 'stage_2' / 'model_best.pth.tar')
            assert Path(args.checkpoint_stage).exists(), f"{args.checkpoint_stage} is not exist!"

            checkpoint = torch.load(args.checkpoint_stage)
            model.load_state_dict(checkpoint['model_state_dict'])
            fc.load_state_dict(checkpoint['fc'])

            state_dim = args.model_dim
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
        else:
            raise ValueError
    elif args.train_method in ['scratch']:
        if args.train_stage == 1:
            pass
        elif args.train_stage == 2:
            if args.checkpoint_stage is None:
                args.checkpoint_stage = str(Path(args.save_dir).parent / 'stage_1' / 'model_best.pth.tar')
            assert Path(args.checkpoint_path).exists(), f"{args.checkpoint_path} is not exist!"

            checkpoint = torch.load(args.checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            fc.load_state_dict(checkpoint['fc'])

            state_dim = args.model_dim
            ppo = rlmil.PPO(dim_patch, state_dim, args.policy_hidden_dim, args.policy_conv,
                            action_std=args.action_std,
                            lr=args.ppo_lr,
                            gamma=args.ppo_gamma,
                            K_epochs=args.K_epochs,
                            action_size=args.num_clusters)
        elif args.train_stage == 3:
            if args.checkpoint_stage is None:
                args.checkpoint_stage = str(Path(args.save_dir).parent / 'stage_2' / 'model_best.pth.tar')
            assert Path(args.checkpoint_stage).exists(), f'{str(args.checkpoint_stage)} is not exists!'

            checkpoint = torch.load(args.checkpoint_stage)
            model.load_state_dict(checkpoint['model_state_dict'])
            fc.load_state_dict(checkpoint['fc'])

            state_dim = args.model_dim
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
    fc = fc.cuda()

    assert model is not None, "creating model failed. "
    print(f"model Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"fc Total params: {sum(p.numel() for p in fc.parameters()) / 1e6:.2f}M")
    return model, fc, ppo


def get_criterion(args):
    if args.loss == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        raise ValueError(f"args.loss error, error value is {args.loss}.")
    return criterion


def get_optimizer(args, model, fc):
    if args.train_stage != 2:
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
        args.epochs = args.ppo_epochs
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
        raise ValueError
    return scheduler


# Train Model Functions ------------------------------------------------------------------------------------------------
def train_CLAM(args, epoch, train_set, model, fc, ppo, memory, criterion, optimizer, scheduler):
    print(f"train_CLAM...")
    length = len(train_set)
    train_set.shuffle()

    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    if args.train_stage == 2:
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()

    progress_bar = tqdm(range(args.num_data))
    feat_list, cluster_list, label_list, step = [], [], [], 0
    batch_idx = 0
    labels_list, outputs_list = [], []
    for data_idx in progress_bar:
        loss_list = []

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

            if args.train_stage != 2:
                outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
                outputs = fc(outputs, restart=True)
            else:
                with torch.no_grad():
                    outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
                    outputs = fc(outputs, restart=True)

            loss = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * result_dict['instance_loss']
            loss_list.append(loss)

            losses[0].update(loss.data.item(), len(feat_list))
            acc = accuracy(outputs, labels, topk=(1,))[0]
            top1[0].update(acc.item(), len(feat_list))

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

                if args.train_stage != 2:
                    outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
                    outputs = fc(outputs, restart=False)
                else:
                    with torch.no_grad():
                        outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
                        outputs = fc(outputs, restart=False)
                loss = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * result_dict[
                    'instance_loss']
                loss_list.append(loss)
                losses[patch_step].update(loss.data.item(), len(feat_list))

                acc = accuracy(outputs, labels, topk=(1,))[0]
                top1[patch_step].update(acc.item(), len(feat_list))

                confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
                reward = confidence - confidence_last
                confidence_last = confidence

                reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                memory.rewards.append(reward)

            loss = sum(loss_list) / args.T
            if args.train_stage != 2:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                ppo.update(memory)
            memory.clear_memory()
            torch.cuda.empty_cache()

            # save the last outputs
            labels_list.append(labels.detach())
            outputs_list.append(outputs.detach())

            feat_list, cluster_list, label_list, step = [], [], [], 0
            batch_idx += 1
            progress_bar.set_description(
                f"Train Epoch: {epoch + 1:2}/{args.epochs:2}. Iter: {batch_idx:3}/{args.eval_step:3}. "
                f"Loss: {losses[-1].avg:.4f}. Acc: {top1[-1].avg:.4f}"
            )
            progress_bar.update()

    progress_bar.close()
    if scheduler is not None and epoch >= args.warmup:
        scheduler.step()

    labels = torch.cat(labels_list)
    outputs = torch.cat(outputs_list)
    acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)

    return losses[-1].avg, acc, auc, precision, recall, f1_score


def test_CLAM(args, test_set, model, fc, ppo, memory, criterion):
    losses = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    model.eval()
    fc.eval()
    with torch.no_grad():
        feat_list, cluster_list, label_list, case_id_list, step = [], [], [], [], 0
        for data_idx, (feat, cluster, label, case_id) in enumerate(test_set):
            feat = feat.unsqueeze(0).to(args.device)
            label = label.unsqueeze(0).to(args.device)
            feat_list.append(feat)
            cluster_list.append(cluster)
            label_list.append(label)
            case_id_list.append(case_id)

        loss_list = []

        labels = torch.cat(label_list)
        action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
        feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)

        outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
        outputs = fc(outputs, restart=True)

        ins_loss = 0
        for r in result_dict:
            ins_loss = ins_loss + r['instance_loss']
        ins_loss = ins_loss / len(feat_list)
        loss = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * ins_loss
        loss_list.append(loss)

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
            outputs, states, result_dict = model(feats, label=labels, instance_eval=True)
            outputs = fc(outputs, restart=False)

            ins_loss = 0
            for r in result_dict:
                ins_loss = ins_loss + r['instance_loss']
            ins_loss = ins_loss / len(feat_list)
            loss = args.bag_weight * criterion(outputs, labels) + (1 - args.bag_weight) * ins_loss
            loss_list.append(loss)
            losses[patch_step].update(loss.data.item(), len(feat_list))

            confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
            reward = confidence - confidence_last
            confidence_last = confidence

            reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
            memory.rewards.append(reward)
        memory.clear_memory()
        acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)
    return losses[-1].avg, acc, auc, precision, recall, f1_score, outputs, labels, case_id_list


def train_DSMIL(args, epoch, train_set, model, fc, ppo, memory, criterion, optimizer, scheduler):
    print(f"train_DSMIL...")
    length = len(train_set)
    train_set.shuffle()

    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    if args.train_stage == 2:
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()

    progress_bar = tqdm(range(args.num_data))
    feat_list, cluster_list, label_list, step = [], [], [], 0
    batch_idx = 0
    labels_list, outputs_list = [], []
    for data_idx in progress_bar:
        loss_list = []

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

            if args.train_stage != 2:
                outputs_ins, outputs, states = model(feats)
                states = torch.mean(states, dim=1)
                outputs_max, _ = torch.max(outputs_ins, 0, keepdim=True)
                outputs = torch.mean(outputs, dim=1)
                outputs = fc(outputs, restart=True)
            else:
                with torch.no_grad():
                    outputs_ins, outputs, states = model(feats)
                    states = torch.mean(states, dim=1)
                    outputs_max, _ = torch.max(outputs_ins, 0, keepdim=True)
                    outputs = torch.mean(outputs, dim=1)
                    outputs = fc(outputs, restart=True)

            loss_max = criterion(outputs_max, labels)

            loss = 0.5 * criterion(outputs, labels) + 0.5 * loss_max
            loss_list.append(loss)

            losses[0].update(loss.data.item(), len(feat_list))
            acc = accuracy(outputs, labels, topk=(1,))[0]
            top1[0].update(acc.item(), len(feat_list))

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

                if args.train_stage != 2:
                    outputs_ins, outputs, states = model(feats)
                    states = torch.mean(states, dim=1)
                    outputs_max, _ = torch.max(outputs_ins, 0, keepdim=True)
                    outputs = torch.mean(outputs, dim=1)
                    outputs = fc(outputs, restart=False)
                else:
                    with torch.no_grad():
                        outputs_ins, outputs, states = model(feats)
                        states = torch.mean(states, dim=1)
                        outputs_max, _ = torch.max(outputs_ins, 0, keepdim=True)
                        outputs = torch.mean(outputs, dim=1)
                        outputs = fc(outputs, restart=False)
                loss_max = criterion(outputs_max, labels)

                loss = 0.5 * criterion(outputs, labels) + 0.5 * loss_max
                loss_list.append(loss)
                losses[patch_step].update(loss.data.item(), len(feat_list))
                acc = accuracy(outputs, labels, topk=(1,))[0]
                top1[patch_step].update(acc.item(), outputs.shape[0])

                confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
                reward = confidence - confidence_last
                confidence_last = confidence

                reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                memory.rewards.append(reward)

            loss = sum(loss_list) / args.T
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

    progress_bar.close()
    if args.train_stage != 2 and scheduler is not None and epoch >= args.warmup:
        scheduler.step()

    labels = torch.cat(labels_list)
    outputs = torch.cat(outputs_list)
    acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)
    return losses[-1].avg, acc, auc, precision, recall, f1_score


def test_DSMIL(args, test_set, model, fc, ppo, memory, criterion):
    losses = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    model.eval()
    fc.eval()
    with torch.no_grad():
        feat_list, cluster_list, label_list, case_id_list, step = [], [], [], [], 0
        for data_idx, (feat, cluster, label, case_id) in enumerate(test_set):
            feat = feat.unsqueeze(0).to(args.device)
            label = label.unsqueeze(0).to(args.device)
            feat_list.append(feat)
            cluster_list.append(cluster)
            label_list.append(label)
            case_id_list.append(case_id)

        loss_list = []
        labels = torch.cat(label_list)
        action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
        feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)
        outputs_ins, outputs, states = model(feats)
        states = torch.mean(states, dim=1)
        outputs_max = []
        for ins in outputs_ins:
            m_ins, _ = torch.max(ins, 0, keepdim=True)
            outputs_max.append(m_ins)
        outputs_max = torch.cat(outputs_max)
        outputs = torch.mean(outputs, dim=1)
        outputs = fc(outputs, restart=True)

        loss_max = criterion(outputs_max, labels)

        loss = 0.5 * criterion(outputs, labels) + 0.5 * loss_max
        loss_list.append(loss)

        losses[0].update(loss.data.item(), outputs.shape[0])

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
            outputs_ins, outputs, states = model(feats)
            states = torch.mean(states, dim=1)
            outputs_max = []
            for ins in outputs_ins:
                m_ins, _ = torch.max(ins, 0, keepdim=True)
                outputs_max.append(m_ins)
            outputs_max = torch.cat(outputs_max)
            outputs = torch.mean(outputs, dim=1)
            outputs = fc(outputs, restart=False)

            loss_max = criterion(outputs_max, labels)

            loss = 0.5 * criterion(outputs, labels) + 0.5 * loss_max
            loss_list.append(loss)
            losses[patch_step].update(loss.data.item(), len(feat_list))

            confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
            reward = confidence - confidence_last
            confidence_last = confidence

            reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
            memory.rewards.append(reward)
        memory.clear_memory()
        acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)
    return losses[-1].avg, acc, auc, precision, recall, f1_score, outputs, labels, case_id_list


def train_ABMIL(args, epoch, train_set, model, fc, ppo, memory, criterion, optimizer, scheduler):
    print(f"train_ABMIL...")
    length = len(train_set)
    train_set.shuffle()

    losses = [AverageMeter() for _ in range(args.T)]
    top1 = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]

    if args.train_stage == 2:
        model.eval()
        fc.eval()
    else:
        model.train()
        fc.train()

    progress_bar = tqdm(range(args.num_data))
    feat_list, cluster_list, label_list, step = [], [], [], 0
    batch_idx = 0
    labels_list, outputs_list = [], []
    for data_idx in progress_bar:
        loss_list = []

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
            if args.train_model_prime and args.train_stage != 2:
                outputs, states = model(feats)
                outputs = fc(outputs, restart=True)
            else:
                with torch.no_grad():
                    outputs, states = model(feats)
                    outputs = fc(outputs, restart=True)

            loss = criterion(outputs, labels)
            loss_list.append(loss)

            losses[0].update(loss.data.item(), len(feat_list))
            acc = accuracy(outputs, labels, topk=(1,))[0]
            top1[0].update(acc.item(), len(feat_list))

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

                if args.train_stage != 2:
                    outputs, states = model(feats)
                    outputs = fc(outputs, restart=False)
                else:
                    with torch.no_grad():
                        outputs, states = model(feats)
                        outputs = fc(outputs, restart=False)
                loss = criterion(outputs, labels)
                loss_list.append(loss)
                losses[patch_step].update(loss.data.item(), len(feat_list))

                acc = accuracy(outputs, labels, topk=(1,))[0]
                top1[patch_step].update(acc.item())

                confidence = torch.gather(F.softmax(outputs.detach(), 1), dim=1, index=labels.view(-1, 1)).view(1, -1)
                reward = confidence - confidence_last
                confidence_last = confidence

                reward_list[patch_step - 1].update(reward.data.mean(), len(feat_list))
                memory.rewards.append(reward)

            loss = sum(loss_list) / args.T
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

    progress_bar.close()
    if args.train_stage != 2 and scheduler is not None and epoch >= args.warmup:
        scheduler.step()

    labels = torch.cat(labels_list)
    outputs = torch.cat(outputs_list)
    acc, auc, precision, recall, f1_score = get_metrics(outputs, labels)

    return losses[-1].avg, acc, auc, precision, recall, f1_score


def test_ABMIL(args, test_set, model, fc, ppo, memory, criterion):
    losses = [AverageMeter() for _ in range(args.T)]
    reward_list = [AverageMeter() for _ in range(args.T - 1)]
    model.eval()
    fc.eval()
    with torch.no_grad():
        feat_list, cluster_list, label_list, case_id_list, step = [], [], [], [], 0
        for data_idx, (feat, cluster, label, case_id) in enumerate(test_set):
            feat = feat.unsqueeze(0).to(args.device)
            label = label.unsqueeze(0).to(args.device)
            feat_list.append(feat)
            cluster_list.append(cluster)
            label_list.append(label)
            case_id_list.append(case_id)

        loss_list = []
        labels = torch.cat(label_list)
        action_sequence = torch.rand((len(feat_list), args.num_clusters), device=feat_list[0].device)
        feats = get_feats(feat_list, cluster_list, action_sequence=action_sequence, feat_size=args.feat_size)

        outputs, states = model(feats)
        outputs = fc(outputs, restart=True)

        loss = criterion(outputs, labels)
        loss_list.append(loss)

        losses[0].update(loss.data.item(), outputs.shape[0])

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
            outputs, states = model(feats)
            outputs = fc(outputs, restart=False)

            loss = criterion(outputs, labels)
            loss_list.append(loss)
            losses[patch_step].update(loss.data.item(), len(feat_list))

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
def train(args, train_set, valid_set, test_set, model, fc, ppo, memory, criterion, optimizer, scheduler, tb_writer):
    # Init variables
    save_dir = args.save_dir
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
                            header=['epoch', 'final_epoch', 'final_loss', 'final_acc', 'final_auc', 'final_precision',
                                    'final_recall', 'final_f1_score'])

    best_model = copy.deepcopy({'state_dict': model.state_dict()})
    early_stop = EarlyStop(max_num_accordance=args.patience) if args.patience is not None else None

    for epoch in range(args.epochs):
        print(f"Training Stage: {args.train_stage}, lr:")
        if optimizer is not None:
            for k, group in enumerate(optimizer.param_groups):
                print(f"group[{k}]: {group['lr']}")

        train_loss, train_acc, train_auc, train_precision, train_recall, train_f1_score = \
            TRAIN[args.arch](args, epoch, train_set, model, fc, ppo, memory, criterion, optimizer, scheduler)
        valid_loss, valid_acc, valid_auc, valid_precision, valid_recall, valid_f1_score, *_ = \
            TEST[args.arch](args, valid_set, model, fc, ppo, memory, criterion, mode='Valid')
        test_loss, test_acc, test_auc, test_precision, test_recall, test_f1_score, *_ = \
            TEST[args.arch](args, test_set, model, fc, ppo, memory, criterion, mode='Test ')

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
            'fc': fc.state_dict(),
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
        results_csv.write_row(
            [epoch + 1, final_epoch, test_loss, test_acc, test_auc, test_precision, test_recall, test_f1_score])

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
        if early_stop is not None:
            early_stop.update((best_valid_loss.best, best_valid_acc.best, best_valid_auc.best))
            if early_stop.is_stop():
                break

    if tb_writer is not None:
        tb_writer.close()

    return best_model


def test(args, test_set, model, fc, ppo, memory, criterion):
    model.eval()
    fc.eval()
    with torch.no_grad():
        loss, acc, auc, precision, recall, f1_score, outputs_tensor, labels_tensor, case_id_list = \
            TEST[args.arch](args, test_set, model, fc, ppo, memory, criterion)
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
    init_seeds(args.seed)

    if args.save_dir is None:
        create_save_dir(args)
    else:
        args.save_dir = str(Path(args.base_save_dir) / args.save_dir)
    args.save_dir = increment_path(Path(args.save_dir), exist_ok=args.exist_ok, sep='_')  # increment run
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    # Dataset
    datasets, dim_patch, train_length = get_datasets(args)
    args.num_data = train_length
    args.eval_step = int(args.num_data / args.batch_size)
    print(f"train_length: {train_length}, epoch_step: {args.num_data}, eval_step: {args.eval_step}")

    # Model, Criterion, Optimizer and Scheduler
    model, fc, ppo = create_model(args, dim_patch)
    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model, fc)
    scheduler = get_scheduler(args, optimizer)

    # Save arguments
    with open(Path(args.save_dir) / 'args.yaml', 'w') as fp:
        yaml.dump(vars(args), fp, sort_keys=False)
    print(args, '\n')

    # TensorBoard
    tb_writer = SummaryWriter(args.save_dir) if args.use_tensorboard else None

    # Start training
    memory = rlmil.Memory()
    best_model = train(args, datasets['train'], datasets['valid'], datasets['test'], model, fc, ppo, memory, criterion,
                       optimizer, scheduler, tb_writer)
    model.module.load_state_dict(best_model['model_state_dict'])
    fc.load_state_dict(best_model['fc'])
    if ppo is not None:
        ppo.policy.load_state_dict(best_model['policy'])
    loss, acc, auc, precision, recall, f1_score, preds = \
        test(args, datasets['test'], model, fc, ppo, memory, criterion)

    # Save results
    preds.to_csv(str(Path(args.save_dir) / 'pred.csv'))
    final_res = pd.DataFrame(columns=['loss', 'acc', 'auc', 'precision', 'recall', 'f1_score'])
    final_res.loc[f'seed{args.seed}'] = [loss, acc, auc, precision, recall, f1_score]
    final_res.to_csv(str(Path(args.save_dir) / 'final_res.csv'))
    print(f'{final_res}\nPredicted Ending.\n')


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str, default='Camelyon16',
                        help="dataset name")
    parser.add_argument('--data_csv', type=str, default='',
                        help="the .csv filepath used")
    parser.add_argument('--data_split_json', type=str, default='/path/to/data_split.json')
    parser.add_argument('--train_data', type=str, default='train', choices=['train', 'train_sub_per10'],
                        help="specify how much data used")
    parser.add_argument('--preload', action='store_true', default=False,
                        help="preload the patch features, default False")
    parser.add_argument('--feat_size', default=1024, type=int,
                        help="the size of selected WSI set. (we recommend 1024 at 20x magnification")
    # Train
    parser.add_argument('--train_method', type=str, default='scratch', choices=['scratch', 'finetune', 'linear'])
    parser.add_argument('--train_stage', default=1, type=int,
                        help="select training stage \
                                  stage-1 : warm-up \
                                  stage-2 : learn to select patches with RL \
                                  stage-3 : finetune")
    parser.add_argument('--T', default=6, type=int,
                        help="maximum length of the sequence of RNNs")
    parser.add_argument('--checkpoint_stage', default=None, type=str,
                        help="path to the stage-1/2 checkpoint (for training stage-2/3)")
    parser.add_argument('--checkpoint_pretrained', type=str, default=None,
                        help='path to the pretrained checkpoint (for finetune and linear)')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'],
                        help="specify the optimizer used, default Adam")
    parser.add_argument('--scheduler', type=str, default=None, choices=[None, 'StepLR', 'CosineAnnealingLR'],
                        help="specify the lr scheduler used, default None")
    parser.add_argument('--batch_size', type=int, default=1,
                        help="the batch size for training")
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--ppo_epochs', type=int, default=10,
                        help="the training epochs for R")
    parser.add_argument('--backbone_lr', default=1e-4, type=float)
    parser.add_argument('--fc_lr', default=1e-4, type=float)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', action='store_true', default=True)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--warmup', default=0, type=float,
                        help="the number of epoch for training without lr scheduler, if scheduler is not None")
    parser.add_argument('--wdecay', default=1e-5, type=float,
                        help="the weight decay of optimizer")
    parser.add_argument('--picked_method', type=str, default='score',
                        help="the metric of pick best model from validation dataset")
    parser.add_argument('--patience', type=int, default=None,
                        help="if the loss not change during `patience` epochs, the training will early stop")

    # Architecture
    parser.add_argument('--arch', default='CLAM_SB', type=str, choices=MODELS, help='model name')
    parser.add_argument('--num_classes', type=int, default=2)
    # Architecture - PPO
    parser.add_argument('--policy_hidden_dim', type=int, default=512)
    parser.add_argument('--policy_conv', action='store_true', default=False)
    parser.add_argument('--action_std', type=float, default=0.5)
    parser.add_argument('--ppo_lr', type=float, default=0.00001)
    parser.add_argument('--ppo_gamma', type=float, default=0.1)
    parser.add_argument('--K_epochs', type=int, default=3)
    # Architecture - Full_layer
    parser.add_argument('--feature_num', type=int, default=512)
    parser.add_argument('--fc_hidden_dim', type=int, default=1024)
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
    parser.add_argument('--base_save_dir', type=str, default='./results')
    parser.add_argument('--save_dir', type=str, default=None,
                        help="specify the save directory to save experiment results, default None."
                             "If not specify, the directory will be create by function create_save_dir(args)")
    parser.add_argument('--save_dir_flag', type=str, default=None,
                        help="append a `string` to the end of save_dir")
    parser.add_argument('--exist_ok', action='store_true', default=False)
    parser.add_argument('--save_model', action='store_true', default=False)
    # Global
    parser.add_argument('--device', default='2',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=985)
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    # Pandas print setting
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    torch.set_num_threads(1)

    # Global variables
    MODELS = ['ABMIL', 'CLAM_SB', 'DSMIL']

    LOSSES = ['CrossEntropyLoss']

    TRAIN = {
        'ABMIL': train_ABMIL,
        'DSMIL': train_DSMIL,
        'CLAM_SB': train_CLAM,
    }
    TEST = {
        'ABMIL': test_ABMIL,
        'DSMIL': test_DSMIL,
        'CLAM_SB': test_CLAM,
    }

    main()
