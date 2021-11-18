# General utils
import glob
import os
import random
import re
import yaml
import json
import csv
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support


def init_seeds(seed=0):
    # # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def setup_seed(seed):
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (file, files)  # assert unique
        return files[0]  # return file


def increment_path(path, exist_ok=True, sep=''):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path


def load_yaml(filename):
    with open(filename) as fp:
        return yaml.load(fp, Loader=yaml.FullLoader)


def dump_json(data_dict, filename):
    with open(filename, 'w', encoding='utf-8') as fp:
        json.dump(data_dict, fp)


def load_json(filename):
    with open(filename) as fp:
        data_dict = json.load(fp)
    return data_dict


class EarlyStop:
    def __init__(self, max_num_accordance=5):
        self.max_num_accordance = max_num_accordance
        self.base_variable = ()
        self.num_accordance = 0

    def update(self, variable):
        if variable == self.base_variable:
            self.num_accordance += 1
        else:
            self.num_accordance = 1
            self.base_variable = variable

    def is_stop(self):
        return self.num_accordance >= self.max_num_accordance


class CSVWriter:
    def __init__(self, filename, header=None, sep=',', append=False):
        self.filename = filename
        self.sep = sep
        if Path(self.filename).exists() and not append:
            os.remove(self.filename)
        if header is not None:
            self.write_row(header)

    def write_row(self, row):
        with open(self.filename, 'a+') as fp:
            csv_writer = csv.writer(fp, delimiter=self.sep)
            csv_writer.writerow(row)

    def write_rows(self, rows):
        with open(self.filename, 'a+') as fp:
            csv_writer = csv.writer(fp, delimiter=self.sep)
            csv_writer.writerows(rows)


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class BestVariable:
    def __init__(self, order='max'):
        self.order = order
        if self.order == 'max':
            self.best = float('-inf')
        elif self.order == 'min':
            self.best = float('inf')
        else:
            raise ValueError(f"")
        self.epoch = 0

    def reset(self):
        if self.order == 'max':
            self.best = float('-inf')
        elif self.order == 'min':
            self.best = float('inf')
        else:
            raise ValueError(f"")
        self.epoch = 0

    def compare(self, val, epoch=None, inplace=False):
        flag = True if (self.order == 'max' and val > self.best) or (self.order == 'min' and val < self.best) else False
        if flag and inplace:
            self.best = val
            if epoch is not None:
                self.epoch = epoch
        return flag


def bootstrap_auc(y_true, y_pred, n_bootstraps=2000, rng_seed=42):
    n_bootstraps = n_bootstraps
    rng_seed = rng_seed
    bootstrapped_scores = []

    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(len(y_pred), size=len(y_pred))
        score = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_scores.append(score)
        # print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
    bootstrapped_scores = np.array(bootstrapped_scores)

    print("AUROC: {:0.3f}".format(roc_auc_score(y_true, y_pred)))
    print("Confidence interval for the AUROC score: [{:0.3f} - {:0.3}]".format(
        np.percentile(bootstrapped_scores, (2.5, 97.5))[0], np.percentile(bootstrapped_scores, (2.5, 97.5))[1]))

    return roc_auc_score(y_true, y_pred), np.percentile(bootstrapped_scores, (2.5, 97.5))


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label >= threshold_optimal] = 1
    this_class_label[this_class_label < threshold_optimal] = 0
    bag_predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average='binary')
    accuracy = 1 - np.count_nonzero(np.array(bag_labels).astype(int) - bag_predictions.astype(int)) / len(bag_labels)
    return accuracy, auc_value, precision, recall, fscore


def acc_auc(outputs, targets):
    with torch.no_grad():
        assert outputs.shape[0] == targets.shape[0]
        bs = targets.shape[0]

        if outputs.shape[1] > 2:
            return multi_class_acc_auc(outputs, targets)

        targets = np.asarray(targets.cpu().numpy(), dtype=int).reshape(-1)
        probs = np.asarray(torch.softmax(outputs, dim=1).cpu().numpy())
        auc = roc_auc_score(targets, probs[:, 1])

        fpr, tpr, threshold = roc_curve(targets, probs[:, 1], pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        preds = np.array(probs[:, 1])
        preds[preds >= threshold_optimal] = 1
        preds[preds < threshold_optimal] = 0
        correct = np.equal(preds, targets)
        acc = sum(correct) / bs

    return acc, auc


def multi_class_acc_auc(outputs, targets):
    with torch.no_grad():
        assert outputs.shape[0] == targets.shape[0]
        bs = targets.shape[0]
        _, preds = torch.max(outputs, dim=1)
        correct = preds.eq(targets)
        acc = sum(correct) / bs

        targets = np.asarray(targets.cpu().numpy(), dtype=int).reshape(-1)
        probs = np.asarray(torch.softmax(outputs, dim=1).cpu().numpy())
        auc = roc_auc_score(targets, probs, multi_class='ovr')

    return acc.item(), auc


def get_metrics(outputs, targets):
    with torch.no_grad():
        assert outputs.shape[0] == targets.shape[0]
        bs = targets.shape[0]
        num_class = outputs.shape[1]
        multi_class = True if num_class > 2 else False

        # ACC
        _, preds = torch.max(outputs, dim=1)
        correct = preds.eq(targets)
        acc = (sum(correct) / bs).item()
        # print(f"targets: {targets}")
        # print(f"preds: {preds}")

        # AUC
        targets = np.asarray(targets.cpu().numpy(), dtype=int).reshape(-1)
        probs = np.asarray(torch.softmax(outputs, dim=1).cpu().numpy())
        if multi_class:
            auc = roc_auc_score(targets, probs, multi_class='ovr')
        else:
            auc = roc_auc_score(targets, probs[:, 1])

        # precision, recall, f1_score
        preds = preds.cpu().numpy()
        if multi_class:
            precision, recall, f1_score, _ = precision_recall_fscore_support(targets, preds, average='macro')
        else:
            precision, recall, f1_score, _ = precision_recall_fscore_support(targets, preds, average='binary')
    return acc, auc, precision, recall, f1_score


def get_score(acc, auc, precision, recall, f1_score):
    return 0.3 * acc + 0.3 * auc + 0.1 * precision + 0.1 * recall + 0.2 * f1_score


# Test -----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
