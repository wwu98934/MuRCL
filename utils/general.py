import os
import re
import csv
import yaml
import json
import glob
import shutil
import random
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support


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
    with open(filename, 'r', encoding='utf-8') as fp:
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


class BestVariable:
    def __init__(self, order='max'):
        self.order = order
        if self.order == 'max':
            self.best = float('-inf')
        elif self.order == 'min':
            self.best = float('inf')
        else:
            raise ValueError
        self.epoch = 0

    def reset(self):
        if self.order == 'max':
            self.best = float('-inf')
        elif self.order == 'min':
            self.best = float('inf')
        else:
            raise ValueError
        self.epoch = 0

    def compare(self, val, epoch=None, inplace=False):
        flag = True if (self.order == 'max' and val > self.best) or (self.order == 'min' and val < self.best) else False
        if flag and inplace:
            self.best = val
            if epoch is not None:
                self.epoch = epoch
        return flag


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


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


# Test -----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
