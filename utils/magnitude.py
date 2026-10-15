"""
Local magnitude weight map calculation (Section III-B, Algorithm 2).

Magnitude quantifies the spectral-spatial heterogeneity of each pixel within
its local window, serving as an unsupervised boundary indicator. The weight
vector satisfies  zeta_X . w_X = 1  (Eq. 1), where the similarity matrix is
zeta_X(i,j) = exp(-d_mix(i,j)) and d_mix = alpha * d_spe + (1-alpha) * d_spa
(Eq. 8-10). The magnitude weight of the center pixel is  W = 1^T zeta^-1 1
(Eq. 11).

This is an OFFLINE precomputation step: run it once per dataset to produce a
magnitude weight map saved as a .npy file, then reference it during training.
"""

import numpy as np
from tqdm import tqdm


def min_max_normalize(data):
    """Band-wise min-max normalization for an (H, W, B) cube."""
    out = np.zeros_like(data, dtype=np.float32)
    for b in range(data.shape[2]):
        ch = data[:, :, b]
        mn, mx = ch.min(), ch.max()
        out[:, :, b] = (ch - mn) / (mx - mn) if mx > mn else 0
    return out


def normalize_2d(data):
    """Min-max normalize a 2D weight map to [0, 1]."""
    mn, mx = data.min(), data.max()
    return (data - mn) / (mx - mn) if mx > mn else np.zeros_like(data)


def compute_local_weight_map(hsi_cube, window_size=7, alpha=0.7, beta=0.3):
    """
    Compute the local magnitude weight map for a full HSI cube.

    Args:
        hsi_cube:   normalized HSI, shape (H, W, B).
        window_size: local window extent ws (Algorithm 2).
        alpha:      spectral-vs-spatial mixing weight (Eq. 10).
        beta:       spatial-distance weight in the mixed distance.
    Returns:
        weight_map: (H, W) magnitude weight map normalized to [0, 1].
    """
    hsi_cube = min_max_normalize(hsi_cube)
    H, W, B = hsi_cube.shape
    weight_map = np.full((H, W), np.nan, dtype=np.float32)
    half_win = window_size // 2

    for i in tqdm(range(H), desc="Computing magnitude weight map"):
        for j in range(W):
            i_start, i_end = max(0, i - half_win), min(H, i + half_win + 1)
            j_start, j_end = max(0, j - half_win), min(W, j + half_win + 1)

            win_h, win_w = i_end - i_start, j_end - j_start
            if win_h < 3 or win_w < 3:
                weight_map[i, j] = 0
                continue

            window = hsi_cube[i_start:i_end, j_start:j_end]          # [wh, ww, B]
            flat = window.reshape(-1, B)
            n_pixels = flat.shape[0]

            ci, cj = i - i_start, j - j_start
            center_idx = ci * win_w + cj

            # Spectral distance: cosine distance (Eq. 8).
            norms = np.linalg.norm(flat, axis=1, keepdims=True)
            valid = norms > 1e-8
            flat_norm = np.where(valid, flat / norms, 0)
            spectral_dist = 1 - np.dot(flat_norm, flat_norm.T)

            # Spatial distance to the center (Eq. 9).
            spatial_dist = np.zeros((n_pixels, n_pixels), dtype=np.float32)
            for p in range(n_pixels):
                r1, c1 = p // win_w, p % win_w
                for q in range(p + 1, n_pixels):
                    r2, c2 = q // win_w, q % win_w
                    d = ((r1 - ci) ** 2 + (c1 - cj) ** 2) ** 0.5 + \
                        ((r2 - ci) ** 2 + (c2 - cj) ** 2) ** 0.5
                    spatial_dist[p, q] = spatial_dist[q, p] = d

            # Mixed distance and similarity (Eq. 10).
            mixed_dist = alpha * spectral_dist + beta * spatial_dist
            Z = np.exp(-mixed_dist)
            ones = np.ones(n_pixels)

            try:
                w = np.linalg.lstsq(Z, ones, rcond=1e-6)[0]
            except Exception:
                w = ones / n_pixels

            weight_map[i, j] = w[center_idx]

    weight_map = np.nan_to_num(weight_map, nan=0.0)
    return normalize_2d(weight_map)
