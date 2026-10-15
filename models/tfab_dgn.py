"""
TFAB-DGN: Topology-Feature-Aware Boundary Dual Graph Network
for Hyperspectral Image Classification.

Reference:
    Jie Qian and Yi Tang, "Topology-Guided Boundary-Aware Feature Learning
    for Hyperspectral Image Classification With Dual Graph Networks,"
    IEEE Transactions on Geoscience and Remote Sensing, 2026.

This file implements the complete TFAB-DGN architecture described in the
paper. The module naming follows the architecture in Section III (Method):

    - SpectralDimensionalityReduction   (Eq. 4-5)
    - SLICSuperpixelSegmentation        (Eq. 6, external utility)
    - MagnitudeWeightCalculation        (Algorithm 2, external utility)
    - MagnitudeNodeScoring              (Eq. 12-14)
    - ExternalGraphNetwork (GCN Block)  (Eq. 17)
    - InternalGraphNetwork (TFAB Block) (Eq. 18-21)
    - PersistentHomologyReweighting     (Eq. 20)
    - FeatureFusion                     (Eq. 22-24)
    - Classifier                        (Eq. 25)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gudhi as gd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 1. Spectral Dimensionality Reduction (Eq. 4-5)
#    Two iterations of 1x1 convolution with BatchNorm and LeakyReLU, which
#    compress the hundreds of spectral channels into C (< B) key bands.
# =============================================================================
class SpectralDimensionalityReduction(nn.Module):
    """Compress the raw HSI bands to C key bands via two 1x1 convolutions."""

    def __init__(self, in_bands, mid_channel):
        super().__init__()
        self.denoise = nn.Sequential(
            nn.BatchNorm2d(in_bands),
            nn.Conv2d(in_bands, mid_channel, kernel_size=(1, 1)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(mid_channel, mid_channel, kernel_size=(1, 1)),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        # x: [H, W, B] -> [1, B, H, W] -> [1, C, H, W] -> [H, W, C]
        out = self.denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))
        out = torch.squeeze(out, 0).permute([1, 2, 0])
        return out


# =============================================================================
# 2. External Graph Convolution (GCN Block, Eq. 17)
#    Standard graph convolution:  H' = sigma(D^-1/2 A~ D^-1/2 H W)
# =============================================================================
class ExternalGraphConvolution(nn.Module):
    """Standard GCN layer used by the external (smooth-region) graph."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU(inplace=True)

    def _sym_norm(self, g):
        # g assumed to already include self-loops (A~ = A + I)
        d = g.sum(1)
        d_hat = torch.diag(torch.pow(d, -0.5))
        return d_hat

    def forward(self, g, h):
        h = self.bn(h)
        g = g + torch.eye(g.shape[0], g.shape[0], device=device, dtype=torch.float32)
        d_hat = self._sym_norm(g)
        h = torch.matmul(torch.matmul(torch.matmul(d_hat, g), d_hat), h)
        h = self.proj(h)
        h = self.act(h)
        return h


# =============================================================================
# 3. Internal Graph Convolution (Eq. 21)
#    GCN used inside the TFAB block, operating on the PH-reweighted adjacency.
# =============================================================================
class InternalGraphConvolution(nn.Module):
    """GCN layer operating on the PH-reweighted internal graph."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_dim)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.LeakyReLU(inplace=True),
        )

    def _sym_norm(self, g):
        d = g.sum(1)
        d_hat = torch.diag(torch.pow(d, -0.5))
        return d_hat

    def forward(self, g, h):
        h = self.bn(h)
        g = g + torch.eye(g.shape[0], g.shape[0], device=device, dtype=torch.float32)
        d_hat = self._sym_norm(g)
        h = torch.matmul(torch.matmul(torch.matmul(d_hat, g), d_hat), h)
        h = self.proj(h)
        return h


# =============================================================================
# 4. Persistent Homology Reweighting (Eq. 18-20)
#    Build a Vietoris-Rips complex from the internal graph, extract 0-D / 1-D
#    persistence, and reweight the adjacency matrix so that edges belonging to
#    stable cycles receive the 1-D persistence value, while other edges receive
#    the max of the 0-D death values of their endpoints.
# =============================================================================
class PersistentHomologyReweighting(nn.Module):
    """
    Recalibrate the internal adjacency matrix with persistent-homology weights.

    Ring (1-D cycle) edges  ->  weight = persistence (death - birth)
    Ordinary edges          ->  weight = mean of the two endpoints' 0-D death
    """

    def __init__(self, max_edge_length=1.1):
        super().__init__()
        self.max_edge_length = max_edge_length

    @staticmethod
    def _norm_matrix(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-12
        x_min, x_max = x.min(), x.max()
        if torch.allclose(x_min, x_max, atol=eps):
            return torch.zeros_like(x)
        return (x - x_min) / (x_max - x_min + eps)

    def pairwise_euclidean_distance(self, x: torch.Tensor) -> torch.Tensor:
        """Normalized pairwise Euclidean distance matrix [N, N]."""
        dist = torch.cdist(x, x, p=2)
        return self._norm_matrix(dist)

    def _to_numpy(self, t):
        return t.detach().cpu().numpy().astype(np.float32) if isinstance(t, torch.Tensor) \
            else t.astype(np.float32)

    def forward(self, adj_matrix, distance_matrix, plot=False, save_path='./persistence_diagram.png'):
        adj_np = self._to_numpy(adj_matrix)
        dist_np = self._to_numpy(distance_matrix)
        num_nodes = adj_np.shape[0]

        # Build the Vietoris-Rips complex (max dimension 2 for H0/H1).
        rips_complex = gd.RipsComplex(distance_matrix=dist_np,
                                      max_edge_length=self.max_edge_length)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        persistence_values = simplex_tree.persistence(
            homology_coeff_field=2, min_persistence=0.0, persistence_dim_max=False
        )

        if plot:
            gd.plot_persistence_diagram(persistence_values, legend=True)
            import matplotlib.pyplot as plt
            plt.savefig(save_path)
            plt.close()

        persistence_pairs = simplex_tree.persistence_pairs()

        # Separate 0-D and 1-D persistence values.
        dim0_persistence, dim1_persistence = [], []
        for dim, (birth, death) in persistence_values:
            if dim == 0:
                dim0_persistence.append((birth, death))
            elif dim == 1:
                dim1_persistence.append((birth, death))

        # Parse 0-D features (connected components): vertex -> death value.
        dim0_features, dim0_idx = [], 0
        for birth_simplex, death_simplex in persistence_pairs:
            if len(birth_simplex) == 1 and (len(death_simplex) == 2 or len(death_simplex) == 0):
                vertex = birth_simplex[0]
                birth, death = dim0_persistence[dim0_idx] if dim0_idx < len(dim0_persistence) \
                    else (0.0, np.inf)
                dim0_features.append((vertex, birth, death))
                dim0_idx += 1

        # Parse 1-D features (cycles): edge -> persistence value.
        dim1_features, dim1_idx = [], 0
        for birth_simplex, death_simplex in persistence_pairs:
            if len(birth_simplex) == 2 and (len(death_simplex) == 3 or len(death_simplex) == 0):
                edge = tuple(sorted(birth_simplex))
                birth, death = dim1_persistence[dim1_idx] if dim1_idx < len(dim1_persistence) \
                    else (0.0, np.inf)
                dim1_features.append((edge, birth, death))
                dim1_idx += 1

        # Node persistence: 0-D death value (infinite -> max_edge_length).
        node_persistence = np.zeros(num_nodes, dtype=np.float32)
        for vertex, birth, death in dim0_features:
            node_persistence[vertex] = self.max_edge_length if np.isinf(death) else death

        edge_persistence = np.zeros_like(adj_np, dtype=np.float32)

        # 1-D cycle edges -> persistence (death - birth).
        for edge, birth, death in dim1_features:
            u, v = edge
            if adj_np[u, v] == 1:
                persistence_value = (self.max_edge_length - birth) if np.isinf(death) \
                    else (death - birth)
                edge_persistence[u, v] = persistence_value
                edge_persistence[v, u] = persistence_value

        # Remaining edges -> mean of the endpoints' 0-D persistence.
        mask = (adj_np == 1) & (edge_persistence == 0)
        triu_mask = np.triu(mask, k=1)
        i_idx, j_idx = np.where(triu_mask)
        avg_values = (node_persistence[i_idx] + node_persistence[j_idx]) / 2
        edge_persistence[i_idx, j_idx] = avg_values
        edge_persistence[j_idx, i_idx] = avg_values

        # Hadamard product: inject persistence weights into the adjacency.
        persistent_adj_np = adj_np * edge_persistence
        persistent_adj = torch.tensor(persistent_adj_np, dtype=torch.float32)
        if isinstance(adj_matrix, torch.Tensor):
            persistent_adj = persistent_adj.to(adj_matrix.device)

        return self._norm_matrix(persistent_adj)


# =============================================================================
# 5. Magnitude Node Scoring (Eq. 12-14)
#    S(v_i) = beta * S_mlp(v_i) + (1 - beta) * S_mag(v_i)
# =============================================================================
class MagnitudeNodeScoring(nn.Module):
    """Score each superpixel node by fusing an MLP score with the magnitude prior."""

    def __init__(self, in_channels, mlp_hidden_dim=64, beta=0.5):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.tensor([beta], requires_grad=True))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, node_features, superpixel_mask, magnitude_map_path):
        # S_mlp: learnable structural score (Eq. 13).
        base_scores = self.mlp(node_features).squeeze()          # [N]
        base_scores = F.normalize(base_scores, dim=0)

        # S_mag: magnitude-prior score (Eq. 14), averaged over the node's pixels.
        magnitude_map = torch.from_numpy(np.load(magnitude_map_path)).float().to(node_features.device)
        num_nodes = node_features.shape[0]
        magnitude_scores = torch.zeros(num_nodes, device=node_features.device)
        for i in range(num_nodes):
            mask = (superpixel_mask == i)
            if mask.sum() > 0:
                magnitude_scores[i] = magnitude_map[mask].mean()

        # Fusion with the learnable beta (Eq. 12).
        node_scores = self.beta * base_scores + (1 - self.beta) * magnitude_scores
        return node_scores


# =============================================================================
# 6. TFAB Block (Internal Graph Network, Eq. 18-21)
#    For each Top-k boundary node: build a pixel-level 4-neighborhood graph,
#    apply a first GCN, reweight the adjacency with PH, then apply a second GCN.
# =============================================================================
class TFABBlock(nn.Module):
    """Topology-Feature-Aware Boundary block for the internal (boundary) graph."""

    def __init__(self, in_channels, out_channels, k=5, ph_save_path='./persistence_diagram.png'):
        super().__init__()
        self.k = k
        self.bn = nn.BatchNorm1d(in_channels)
        self.gcn1 = InternalGraphConvolution(in_channels, out_channels)
        self.gcn2 = InternalGraphConvolution(in_channels, out_channels)
        self.ph_reweight = PersistentHomologyReweighting()
        self.ph_save_path = ph_save_path

    def _build_pixel_adjacency(self, pixel_coords, H, W):
        """Build the pixel-level 4-neighborhood adjacency matrix (vectorized)."""
        num_pixels = pixel_coords.shape[0]
        if num_pixels == 0:
            return torch.zeros((0, 0), device=device)

        directions = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]],
                                  dtype=torch.long, device=device)
        coords = pixel_coords.unsqueeze(1)                 # [P, 1, 2]
        neighbor_coords = coords + directions              # [P, 4, 2]

        coord_set = {(h.item(), w.item()): i for i, (h, w) in enumerate(pixel_coords)}
        adj = torch.zeros((num_pixels, num_pixels), device=device)

        for i in range(num_pixels):
            for j in range(4):
                nh, nw = neighbor_coords[i, j]
                if 0 <= nh < H and 0 <= nw < W:
                    key = (nh.item(), nw.item())
                    if key in coord_set:
                        adj[i, coord_set[key]] = 1.0
        return adj

    def forward(self, node_scores, superpixel_mask, image_features):
        """
        Args:
            node_scores:     [N] node scores.
            superpixel_mask: [H, W] segmentation matrix (pixel -> node index).
            image_features:  [H, W, C] pixel-level features.
        Returns:
            updated_features: list of [num_pixels, C] tensors for Top-k nodes.
            topk_indices:     Top-k node indices.
        """
        H, W = superpixel_mask.shape
        topk_indices = torch.topk(node_scores, self.k).indices

        updated_features = []
        for node_idx in topk_indices:
            mask = (superpixel_mask == node_idx)
            pixel_coords = torch.nonzero(mask, as_tuple=False)        # [P, 2] (h, w)
            if len(pixel_coords) == 0:
                continue

            pixel_features = image_features[pixel_coords[:, 0], pixel_coords[:, 1]]  # [P, C]

            # 4-neighborhood internal adjacency.
            adj = self._build_pixel_adjacency(pixel_coords, H, W)

            # First GCN.
            x1 = self.gcn1(adj, pixel_features)

            # PH-reweighted adjacency (Eq. 18-20).
            distance_matrix = self.ph_reweight.pairwise_euclidean_distance(self.bn(x1))
            weighted_adj = self.ph_reweight(adj, distance_matrix,
                                            plot=False, save_path=self.ph_save_path)

            # Second GCN (Eq. 21).
            x2 = self.gcn2(weighted_adj, x1)
            updated_features.append(x2)

        return updated_features, topk_indices


# =============================================================================
# 7. Internal Graph Decoder (Eq. 23)
#    Write the internal-graph features back to their pixel positions.
# =============================================================================
class InternalGraphDecoder(nn.Module):
    """Decode the internal-graph features back onto the pixel-level feature map."""

    def forward(self, internal_features, topk_indices, superpixel_mask, feature_map):
        updated = feature_map.clone()
        for i, node_idx in enumerate(topk_indices):
            pixel_features = internal_features[i]
            mask = (superpixel_mask == node_idx)
            node_pixel_indices = mask.view(-1).nonzero().squeeze()

            if len(node_pixel_indices) != pixel_features.shape[0]:
                raise RuntimeError(
                    f"Node {node_idx}: pixel count {len(node_pixel_indices)} != "
                    f"feature count {pixel_features.shape[0]}."
                )
            updated[node_pixel_indices] = pixel_features
        return updated


# =============================================================================
# 8. Spectral-Spatial Convolution (SSConv)
#    Point-wise + depth-wise convolution used after feature fusion.
# =============================================================================
class SpectralSpatialConv(nn.Module):
    """Spectral-spatial convolution (1x1 point conv + depth conv)."""

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.point_conv = nn.Conv2d(in_ch, out_ch // 2, kernel_size=1, bias=False)
        self.depth_conv = nn.Conv2d(out_ch // 2, out_ch, kernel_size=kernel_size,
                                    padding=kernel_size // 2)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        out = self.bn(x)
        out = self.point_conv(out)
        out = self.act1(out)
        out = self.depth_conv(out)
        out = self.act2(out)
        return out


# =============================================================================
# 9. TFAB-DGN (main model)
# =============================================================================
class TFAB_DGN(nn.Module):
    """
    Topology-Feature-Aware Boundary Dual Graph Network.

    Args:
        in_bands:     original number of spectral bands B.
        mid_channel:  reduced feature dimension C (e.g. 50 / 40).
        n_classes:    number of land-cover classes.
        Q:            pixel-to-superpixel mapping matrix [H*W, N].
        Seg:          segmentation matrix [H, W] (pixel -> node index).
        top_k:        number of boundary nodes selected per layer.
        magnitude_path: path to the precomputed magnitude weight map (.npy).
        ph_save_path: path to save the persistence diagram (visualization).
        data_name:    dataset name (for logging).
    """

    def __init__(self, in_bands, mid_channel, n_classes, Q, Seg, top_k,
                 magnitude_path, ph_save_path, data_name):
        super().__init__()
        self.in_bands = int(in_bands)
        self.n_classes = int(n_classes)
        self.mid_channel = int(mid_channel)
        self.top_k = int(top_k)
        self.data_name = str(data_name)

        self.Q = Q
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True) + 1e-12)
        self.Seg = Seg

        self.magnitude_path = str(magnitude_path)
        self.ph_save_path = str(ph_save_path)

        # Learnable multi-scale fusion weights (Eq. 24).
        self.alpha_1 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))
        self.alpha_2 = torch.nn.Parameter(torch.tensor([0.5], requires_grad=True))

        # Module 1: spectral dimensionality reduction (Eq. 5).
        self.spectral_reduction = SpectralDimensionalityReduction(self.in_bands, self.mid_channel)

        # Module 2: external graph network (3 GCN layers, Eq. 17).
        self.ext_gcn1 = ExternalGraphConvolution(self.mid_channel, self.mid_channel)
        self.ext_gcn2 = ExternalGraphConvolution(self.mid_channel, self.mid_channel)
        self.ext_gcn3 = ExternalGraphConvolution(self.mid_channel, self.mid_channel)

        # Module 3: magnitude node scoring (Eq. 12-14).
        self.scoring1 = MagnitudeNodeScoring(self.mid_channel, mlp_hidden_dim=64, beta=0.5)
        self.scoring2 = MagnitudeNodeScoring(self.mid_channel, mlp_hidden_dim=64, beta=0.5)

        # Module 4: internal graph network (TFAB block, Eq. 18-21).
        self.tfab_block = TFABBlock(self.mid_channel, self.mid_channel,
                                    k=self.top_k, ph_save_path=self.ph_save_path)

        # Module 5: internal graph decoder (Eq. 23).
        self.internal_decoder = InternalGraphDecoder()

        # Module 6: post-fusion CNN (spectral-spatial conv).
        self.conv1 = SpectralSpatialConv(self.mid_channel, self.mid_channel, kernel_size=5)
        self.conv2 = SpectralSpatialConv(self.mid_channel, self.mid_channel, kernel_size=5)

        # Module 7: classifier (Eq. 25).
        self.classifier = nn.Linear(self.mid_channel, self.n_classes)

    def forward(self, A, x):
        height, width, _ = x.shape   # x: [H, W, B]

        # ---- Step 1: spectral dimensionality reduction (Eq. 4-5) ----
        clean_x = self.spectral_reduction(x)                       # [H, W, C]
        clean_x_flatten = clean_x.reshape([height * width, -1])    # [H*W, C]

        # ---- Step 2: node features from superpixel mapping (Eq. 7) ----
        superpixel_features = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # [N, C]

        # ---- Step 3: dual graph network ----
        # External graph (smooth regions), Eq. 17.
        h1 = self.ext_gcn1(A, superpixel_features)

        # First internal-graph pass: score -> TFAB block -> decode.
        scores1 = self.scoring1(h1, self.Seg, self.magnitude_path)
        int_feat1, topk1 = self.tfab_block(scores1, self.Seg, clean_x)

        h2 = self.ext_gcn2(A, h1)

        scores2 = self.scoring2(h2, self.Seg, self.magnitude_path)
        int_feat2, topk2 = self.tfab_block(scores2, self.Seg, clean_x)

        h3 = self.ext_gcn3(A, h2)

        # ---- Step 4: feature fusion (Eq. 22-24) ----
        ext_pixel1 = torch.matmul(self.Q, h2)          # [H*W, C]
        ext_pixel2 = torch.matmul(self.Q, h3)          # [H*W, C]

        ext_map1 = torch.sigmoid(ext_pixel1)
        ext_map2 = torch.sigmoid(ext_pixel2)

        fused1 = self.internal_decoder(int_feat1, topk1, self.Seg, ext_map1)
        fused2 = self.internal_decoder(int_feat2, topk2, self.Seg, ext_map2)

        # Multi-scale weighted fusion (Eq. 24).
        gcn_result = self.alpha_1 * fused1 + (1 - self.alpha_1) * fused2   # [H*W, C]

        # ---- Step 5: post-fusion CNN + residual ----
        cnn_in1 = gcn_result.reshape([height, width, -1]).permute([2, 0, 1]).unsqueeze(0)
        cnn_out1 = self.conv1(cnn_in1)                                    # [1, C, H, W]
        cnn_out1 = cnn_out1.squeeze(0).permute([1, 2, 0]).reshape([height * width, -1])

        res = self.alpha_2 * cnn_out1 + (1 - self.alpha_2) * clean_x_flatten

        cnn_in2 = res.reshape([height, width, -1]).permute([2, 0, 1]).unsqueeze(0)
        cnn_out2 = self.conv2(cnn_in2)                                    # [1, C, H, W]
        cnn_out = cnn_out2.squeeze(0).permute([1, 2, 0]).reshape([height * width, -1])

        # ---- Step 6: classification (Eq. 25) ----
        logits = self.classifier(cnn_out)                                  # [H*W, n_classes]
        y = F.softmax(logits, -1)

        return y, cnn_out, topk2, scores2
