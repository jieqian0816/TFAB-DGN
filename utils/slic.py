"""
SLIC superpixel segmentation and graph initialization (Section III-A).

Converts the HSI into a graph representation:
    - SLIC segments the image into superpixels (Eq. 6).
    - S: superpixel spectral-mean features.
    - Q: pixel-to-superpixel mapping matrix.
    - Seg: segmentation matrix (pixel -> node index).
    - A: superpixel adjacency matrix (Pearson-correlation edge weights).
"""

import math

import numpy as np
from skimage.segmentation import slic


def _segments_label_process(labels):
    """Re-index segment labels to be contiguous 0..K-1."""
    labels = np.array(labels, np.int64)
    h, w = labels.shape
    unique = list(set(np.reshape(labels, [-1]).tolist()))
    mapping = {old: new for new, old in enumerate(unique)}
    new_labels = labels.copy()
    for i in range(h):
        for j in range(w):
            new_labels[i, j] = mapping[new_labels[i, j]]
    return new_labels


class SLICGraph:
    """Build the graph (Q, S, Seg, A) from the HSI via SLIC."""

    def __init__(self, hsi, n_segments=1000, compactness=1, max_iter=20, sigma=1,
                 min_size_factor=0.1, max_size_factor=2):
        self.hsi = hsi
        self.n_segments = n_segments
        self.compactness = compactness
        self.max_iter = max_iter
        self.sigma = sigma
        self.min_size_factor = min_size_factor
        self.max_size_factor = max_size_factor

    def segment(self):
        """Run SLIC and build the (Q, S, Seg) triple (Eq. 6, 7)."""
        img = self.hsi
        h, w, d = img.shape

        segments = slic(
            img, n_segments=self.n_segments, compactness=self.compactness,
            max_num_iter=self.max_iter, convert2lab=False, sigma=self.sigma,
            enforce_connectivity=True, min_size_factor=self.min_size_factor,
            max_size_factor=self.max_size_factor, slic_zero=False,
        )

        if segments.max() + 1 != len(set(np.reshape(segments, [-1]).tolist())):
            segments = _segments_label_process(segments)

        self.segments = segments
        self.superpixel_count = segments.max() + 1

        segments_flat = np.reshape(segments, [-1])
        S = np.zeros([self.superpixel_count, d], dtype=np.float32)
        Q = np.zeros([w * h, self.superpixel_count], dtype=np.float32)
        x = np.reshape(img, [-1, d])

        for i in range(self.superpixel_count):
            idx = np.where(segments_flat == i)[0]
            S[i] = np.sum(x[idx], 0) / len(idx)
            Q[idx, i] = 1

        self.S, self.Q = S, Q
        return Q, S, segments

    def build_adjacency(self, sigma=10.0):
        """Build the adjacency matrix A using Pearson-correlation weights (Eq. 6)."""
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)
        h, w = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max == sub_min or A[sub_max, sub_min] != 0:
                    continue

                p1 = self.S[sub_max]
                p2 = self.S[sub_min]
                p1_c = p1 - np.mean(p1)
                p2_c = p2 - np.mean(p2)
                n1 = np.linalg.norm(p1_c)
                n2 = np.linalg.norm(p2_c)
                if n1 == 0 or n2 == 0:
                    continue
                # Pearson correlation -> exp kernel.
                val = np.exp(-0.1 * np.dot(p1_c, p2_c) / (n1 * n2) / (sigma ** 2))
                A[sub_max, sub_min] = A[sub_min, sub_max] = val
        return A

    def run(self, scale):
        """High-level entry: SLIC with the given scale -> (Q, S, A, Seg)."""
        n_segments_init = self.hsi.shape[0] * self.hsi.shape[1] / scale
        self.n_segments = int(n_segments_init)
        Q, S, Seg = self.segment()
        A = self.build_adjacency(sigma=10.0)
        return Q, S, A, Seg
