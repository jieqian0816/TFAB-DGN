"""
Utility functions for sample partitioning, evaluation, and classification-map
visualization.
"""

import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import spectral as spy
from sklearn import metrics


# -----------------------------------------------------------------------------
# Sample partitioning (Section IV: 30 samples per class, or 15 if fewer exist)
# -----------------------------------------------------------------------------
def get_samples_gt(seed, gt, class_count, train_num, samples_type='same_num'):
    """Randomly select training samples and split the rest into val/test.

    Args:
        seed:          random seed.
        gt:            2D ground-truth label map (background = 0).
        class_count:   number of classes.
        train_num:     number of training samples per class.
        samples_type:  'same_num' (fixed count) or 'ratio' (fraction).
    """
    random.seed(seed)
    height, width = gt.shape
    gt_reshape = np.reshape(gt, [-1])
    train_rand_idx = []

    for i in range(class_count):
        idx = np.where(gt_reshape == i + 1)[-1]
        n = len(idx)
        if samples_type == 'ratio':
            rand_idx = random.sample(list(range(n)), int(np.ceil(n * train_num)))
        else:
            real = min(int(train_num), n)
            if real >= np.sum(gt == (i + 1)):
                real = 15
            rand_idx = random.sample(list(range(n)), min(real, n))
        train_rand_idx.append(idx[rand_idx])

    train_data_index = []
    for arr in train_rand_idx:
        train_data_index.extend(arr.tolist())
    train_data_index = set(train_data_index)

    all_data_index = set(range(len(gt_reshape)))
    background_idx = set(np.where(gt_reshape == 0)[-1].tolist())
    test_data_index = list(all_data_index - train_data_index - background_idx)

    val_data_index = random.sample(test_data_index, class_count)
    val_data_index = set(val_data_index)
    test_data_index = [i for i in test_data_index if i not in val_data_index]

    train_gt = np.zeros(gt_reshape.shape)
    test_gt = np.zeros(gt_reshape.shape)
    val_gt = np.zeros(gt_reshape.shape)
    for i in train_data_index:
        train_gt[i] = gt_reshape[i]
    for i in test_data_index:
        test_gt[i] = gt_reshape[i]
    for i in val_data_index:
        val_gt[i] = gt_reshape[i]

    train_gt = np.reshape(train_gt, [height, width])
    test_gt = np.reshape(test_gt, [height, width])
    val_gt = np.reshape(val_gt, [height, width])
    return train_gt, test_gt, val_gt


def gt_to_one_hot(gt, class_count):
    """Convert a 2D label map to a one-hot (H, W, class_count) array."""
    h, w = gt.shape
    one_hot = np.zeros([h, w, class_count], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if gt[i, j] != 0:
                one_hot[i, j, int(gt[i, j]) - 1] = 1
    return one_hot


def build_label_mask(gt, class_count):
    """Build a [H*W, class_count] mask marking labeled pixels."""
    h, w = gt.shape
    flat = np.reshape(gt, [-1])
    mask = np.zeros([h * w, class_count], dtype=np.float32)
    mask[flat != 0] = 1.0
    return mask


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate(predict, gt, gt_onehot, class_count, height, width):
    """Compute OA, AA, Kappa, macro-F1, and per-class accuracy.

    Args:
        predict:   softmax output [H*W, class_count].
        gt:        flattened ground-truth labels [H*W].
        gt_onehot: one-hot labels [H*W, class_count].
    """
    available = (gt != 0).float()
    count = available.sum()
    correct = (torch_argmax(predict) == torch_argmax(gt_onehot)).float() * available
    oa = correct.sum() / count

    output = predict.detach().cpu().numpy()
    gt_np = gt.detach().cpu().numpy()
    pred = np.argmax(output, axis=-1) + 1

    count_per = np.zeros(class_count)
    correct_per = np.zeros(class_count)
    tp = np.zeros(class_count)
    fn = np.zeros(class_count)
    fp = np.zeros(class_count)

    for x in range(len(gt_np)):
        if gt_np[x] != 0:
            true_c = int(gt_np[x] - 1)
            pred_c = int(pred[x] - 1)
            count_per[true_c] += 1
            if pred[x] == gt_np[x]:
                correct_per[true_c] += 1
                tp[true_c] += 1
            else:
                fn[true_c] += 1
                fp[pred_c] += 1

    acc_per = correct_per / (count_per + 1e-10)
    aa = np.average(acc_per)
    recall = tp / (tp + fn + 1e-10)
    precision = tp / (tp + fp + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # Kappa.
    pred_map = np.reshape(pred, [height, width])
    gt_map = np.reshape(gt_np, [height, width])
    pred_list, real_list = [], []
    for i in range(height):
        for j in range(width):
            if gt_map[i, j] != 0:
                pred_list.append(pred_map[i, j])
                real_list.append(gt_map[i, j])
    kappa = metrics.cohen_kappa_score(
        np.array(pred_list).astype(np.int16), np.array(real_list).astype(np.int16))

    return oa, aa, kappa, np.mean(f1), acc_per, f1


def torch_argmax(x):
    import torch
    return torch.argmax(x, dim=1)


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
def draw_classification_map(label, name, scale=4.0, dpi=400, segments=None):
    """Save a classification map to '<name>.png'."""
    fig, ax = plt.subplots()
    spy.imshow(classes=np.array(label).astype(np.int16), fignum=fig.number, cmap='jet')
    if segments is not None:
        for mk in segments:
            plt.contour(mk, colors='white', linewidths=0.1, alpha=1)
    ax.set_axis_off()
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.savefig(name + '.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    plt.close(fig)
