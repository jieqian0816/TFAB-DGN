"""
TFAB-DGN training / evaluation entry point.

Example:
    python main.py --dataset IP --epochs 200 --seed 0

Dataset configurations follow the paper (Section IV-A):
    IP  (Indian Pines):   30/class, scale 60,  mid_dim 50, top_k 5, 200 epochs
    PU  (Pavia Univ.):    30/class, scale 150, mid_dim 50, top_k 5, 150 epochs
    SA  (Salinas):        30/class, scale 300, mid_dim 50, top_k 3, 150 epochs
    LK  (WHU-Hi-LongKou): 30/class, scale 450, mid_dim 40, top_k 3, 150 epochs
"""

import argparse
import os
import time

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from models.tfab_dgn import TFAB_DGN
from utils.slic import SLICGraph
from utils.magnitude import compute_local_weight_map
from utils.metrics import (get_samples_gt, gt_to_one_hot, build_label_mask,
                           evaluate, draw_classification_map)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Dataset registry
# -----------------------------------------------------------------------------
DATASETS = {
    "IP": {
        "name": "indian_",
        "data": "datasets/Indian_pines/Indian_pines_corrected.mat",
        "data_key": "indian_pines_corrected",
        "gt": "datasets/Indian_pines/Indian_pines_gt.mat",
        "gt_key": "indian_pines_gt",
        "class_count": 16, "lr": 0.0005, "epochs": 200,
        "scale": 60, "mid_dim": 50, "top_k": 5,
    },
    "PU": {
        "name": "paviaU_",
        "data": "datasets/PaviaU/PaviaU.mat",
        "data_key": "paviaU",
        "gt": "datasets/PaviaU/PaviaU_gt.mat",
        "gt_key": "paviaU_gt",
        "class_count": 9, "lr": 5e-4, "epochs": 150,
        "scale": 150, "mid_dim": 50, "top_k": 5,
    },
    "SA": {
        "name": "salinas_",
        "data": "datasets/Salinas/Salinas_corrected.mat",
        "data_key": "salinas_corrected",
        "gt": "datasets/Salinas/Salinas_gt.mat",
        "gt_key": "salinas_gt",
        "class_count": 16, "lr": 5e-4, "epochs": 150,
        "scale": 300, "mid_dim": 50, "top_k": 3,
    },
    "LK": {
        "name": "WHU_Hi_LongKou_",
        "data": "datasets/WHU-Hi/WHU-Hi-LongKou/WHU_Hi_LongKou.mat",
        "data_key": "WHU_Hi_LongKou",
        "gt": "datasets/WHU-Hi/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat",
        "gt_key": "WHU_Hi_LongKou_gt",
        "class_count": 9, "lr": 5e-4, "epochs": 150,
        "scale": 450, "mid_dim": 40, "top_k": 3,
    },
}


def min_max_normalize(data):
    """Band-wise min-max normalization (Eq. 4)."""
    out = np.zeros_like(data, dtype=np.float32)
    for b in range(data.shape[2]):
        ch = data[:, :, b]
        mn, mx = ch.min(), ch.max()
        out[:, :, b] = (ch - mn) / (mx - mn + 1e-12) if mx > mn else 0
    return out


def compute_loss(predict, label_onehot, label_mask):
    """Masked cross-entropy loss (Eq. 25)."""
    ce = -torch.mul(label_onehot, torch.log(predict + 1e-12))
    ce = torch.mul(ce, label_mask)
    return torch.sum(ce)


def ensure_magnitude_map(dataset_key, data, out_dir):
    """Precompute (or load) the magnitude weight map for the dataset."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"magnitude_map_{dataset_key}.npy")
    if os.path.exists(path):
        print(f"[magnitude] loading precomputed map: {path}")
        return path
    print(f"[magnitude] computing weight map for {dataset_key} ...")
    wmap = compute_local_weight_map(data, window_size=7, alpha=0.7, beta=0.3)
    np.save(path, wmap)
    print(f"[magnitude] saved to {path}")
    return path


def run(args):
    cfg = DATASETS[args.dataset]
    os.makedirs(args.output_dir, exist_ok=True)
    result_dir = os.path.join(args.output_dir, cfg["name"])
    os.makedirs(result_dir, exist_ok=True)

    # ---- load data ----
    data = sio.loadmat(cfg["data"])[cfg["data_key"]]
    gt = sio.loadmat(cfg["gt"])[cfg["gt_key"]]
    data = min_max_normalize(data.astype(np.float32))

    height, width, bands = data.shape
    class_count = cfg["class_count"]

    # ---- magnitude weight map ----
    mag_path = ensure_magnitude_map(args.dataset, data, os.path.join(args.output_dir, "magnitude"))

    ph_path = os.path.join(result_dir, "persistence_diagram.png")

    # ---- sample partitioning ----
    train_gt, test_gt, val_gt = get_samples_gt(args.seed, gt, class_count, 30, 'same_num')

    train_oh = gt_to_one_hot(train_gt, class_count).reshape(-1, class_count).astype(np.float32)
    test_oh = gt_to_one_hot(test_gt, class_count).reshape(-1, class_count).astype(np.float32)
    val_oh = gt_to_one_hot(val_gt, class_count).reshape(-1, class_count).astype(np.float32)

    train_mask = build_label_mask(train_gt, class_count)
    test_mask = build_label_mask(test_gt, class_count)
    val_mask = build_label_mask(val_gt, class_count)

    # ---- SLIC graph initialization ----
    slicer = SLICGraph(data)
    Q, S, A, Seg = slicer.run(scale=cfg["scale"])
    print(f"[graph] superpixels: {A.shape[0]}, edges: {int((A != 0).sum() / 2)}")

    draw_classification_map(gt, os.path.join(result_dir, cfg["name"] + "gt"))

    A_t = torch.tensor(A, dtype=torch.float32, device=device)
    Q_t = torch.tensor(Q, dtype=torch.float32, device=device)
    Seg_t = torch.tensor(Seg, dtype=torch.float32, device=device)

    train_gt_f = torch.from_numpy(np.reshape(train_gt, [-1]).astype(np.float32)).to(device)
    test_gt_f = torch.from_numpy(np.reshape(test_gt, [-1]).astype(np.float32)).to(device)
    val_gt_f = torch.from_numpy(np.reshape(val_gt, [-1]).astype(np.float32)).to(device)

    train_oh_t = torch.from_numpy(train_oh).to(device)
    test_oh_t = torch.from_numpy(test_oh).to(device)
    val_oh_t = torch.from_numpy(val_oh).to(device)

    train_mask_t = torch.from_numpy(train_mask).to(device)
    test_mask_t = torch.from_numpy(test_mask).to(device)
    val_mask_t = torch.from_numpy(val_mask).to(device)

    net_input = torch.from_numpy(data.astype(np.float32)).to(device)

    # ---- build model ----
    net = TFAB_DGN(bands, cfg["mid_dim"], class_count, Q_t, Seg_t, cfg["top_k"],
                   mag_path, ph_path, cfg["name"]).to(device)

    print(f"[model] parameters: {sum(p.numel() for p in net.parameters()) / 1e3:.3f} K")

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    # ---- training loop (mirrors the original implementation: epochs+1 iterations,
    #      evaluate every 10 steps, save on lower val loss OR higher val OA) ----
    best_loss = 99999.0
    best_val_oa = 0.0
    best_state = None
    net.train()

    for epoch in range(cfg["epochs"] + 1):
        optimizer.zero_grad()
        output, *_ = net(A_t, net_input)
        loss = compute_loss(output, train_oh_t, train_mask_t)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                output, *_ = net(A_t, net_input)
                val_oa, *_ = evaluate(output, val_gt_f, val_oh_t, class_count, height, width)
                val_loss = compute_loss(output, val_oh_t, val_mask_t).item()
            if val_loss < best_loss or val_oa > best_val_oa:
                best_loss = val_loss
                best_val_oa = val_oa
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            print(f"Epoch {epoch:3d}/{cfg['epochs']} | loss={loss.item():.4f} | val loss={val_loss:.4f} | val OA={val_oa:.4f}")
            net.train()

    # ---- evaluation ----
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        output, *_ = net(A_t, net_input)
        oa, aa, kappa, macro_f1, acc_per, f1_per = evaluate(
            output, test_gt_f, test_oh_t, class_count, height, width)

    print("\n========== Final Results ==========")
    print(f"OA={oa:.4f}  AA={aa:.4f}  Kappa={kappa:.4f}  macro-F1={macro_f1:.4f}")
    print("Per-class accuracy:", np.round(acc_per, 4))
    print("===================================")

    # ---- save model + classification map ----
    torch.save(best_state, os.path.join(result_dir, f"{cfg['name']}best_model.pth"))

    classification_map = torch.argmax(output, 1).reshape(height, width).cpu().numpy() + 1
    draw_classification_map(classification_map,
                            os.path.join(result_dir, f"{cfg['name']}{oa:.4f}"))

    with open(os.path.join(result_dir, f"{cfg['name']}_results.txt"), "w") as f:
        f.write(f"OA={oa:.4f}\nAA={aa:.4f}\nKappa={kappa:.4f}\nmacro-F1={macro_f1:.4f}\n")
        f.write(f"per-class accuracy: {np.round(acc_per, 4)}\n")
        f.write(f"per-class F1: {np.round(f1_per, 4)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TFAB-DGN training")
    parser.add_argument("--dataset", type=str, default="IP", choices=["IP", "PU", "SA", "LK"],
                        help="dataset name")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs")
    parser.add_argument("--output_dir", type=str, default="./results", help="output directory")
    args = parser.parse_args()

    if args.epochs is not None:
        DATASETS[args.dataset]["epochs"] = args.epochs

    run(args)
