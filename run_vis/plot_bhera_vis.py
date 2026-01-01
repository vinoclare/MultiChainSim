#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot BHERA visualization artifacts exported by run_bhera_vis.py.

Generates THREE plots (policy entropy removed):
  1) Latent visualization (PCA/t-SNE, 1D/2D)
  2) q(sparse) probability trajectory (mean±std + GT label + F1@thr)
  3) Action visualization (raw / PCA / t-SNE, 1D/2D)

Per user requirements:
  - 2D plots are scatter-only (NO connecting lines).
  - 2D points are colored by DENSE vs SPARSE (not by timestep gradient).
  - Policy entropy plot is removed.
  - t-SNE/PCA embedding uses ONLY the feature vectors (latent/action), and does NOT append any timestep feature.

Usage:
  python plot_bhera_vis.py --log_dir path/to/tb_log_dir
  python plot_bhera_vis.py --npz_path path/to/vis_last100.npz --out_dir path/to/save

Notes:
  - For 1D embedding with t-SNE, we embed all valid (episode,timestep) points, then compute
    per-timestep mean±std across episodes (so the curve is still timestep-indexed, but timestep
    is NOT an input feature to the embedding).
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def find_latest_npz(log_dir: Path) -> Path:
    cands = list(log_dir.glob("vis_last*.npz"))
    if not cands:
        raise FileNotFoundError(f"No vis_last*.npz found in: {log_dir}")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def make_sparse_label(T: int, sparse_segments):
    y = np.zeros((T,), dtype=np.int64)
    if sparse_segments is None:
        return y
    for seg in sparse_segments:
        if seg is None or len(seg) < 2:
            continue
        s, e = int(seg[0]), int(seg[1])
        s = max(0, min(T, s))
        e = max(0, min(T, e))
        if e > s:
            y[s:e] = 1
    return y


def segments_from_label(y: np.ndarray):
    y = np.asarray(y).astype(np.int64).reshape(-1)
    T = y.shape[0]
    segs = []
    in_one = False
    s = 0
    for t in range(T):
        if y[t] == 1 and not in_one:
            in_one = True
            s = t
        if y[t] == 0 and in_one:
            segs.append((s, t))
            in_one = False
    if in_one:
        segs.append((s, T))
    return segs


def shade_sparse(ax, sparse_label: np.ndarray, alpha: float = 0.12):
    for s, e in segments_from_label(sparse_label):
        ax.axvspan(s, e, alpha=alpha)


def to_prob(x: np.ndarray) -> np.ndarray:
    """
    Convert logits or raw scores to probability in [0,1].
    If x already looks like a probability, keep it.
    """
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.full_like(x, np.nan, dtype=np.float32)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if mn >= -1e-3 and mx <= 1.0 + 1e-3:
        return np.clip(x, 0.0, 1.0).astype(np.float32)
    x_clip = np.clip(x, -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-x_clip))).astype(np.float32)


def f1_at_threshold(q_et: np.ndarray, y_t: np.ndarray, mask_et: np.ndarray, thr: float = 0.5) -> float:
    q = np.asarray(q_et, dtype=np.float32)
    mask = (np.asarray(mask_et) > 0.5) & np.isfinite(q)
    y = np.asarray(y_t, dtype=np.int64).reshape(1, -1)
    y = np.broadcast_to(y, q.shape)

    if not mask.any():
        return float("nan")

    qv = q[mask]
    yv = y[mask].astype(np.int64)
    pred = (qv >= thr).astype(np.int64)

    tp = np.sum((pred == 1) & (yv == 1))
    fp = np.sum((pred == 1) & (yv == 0))
    fn = np.sum((pred == 0) & (yv == 1))
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    return float(2 * precision * recall / (precision + recall + 1e-8))


def flatten_points(x_etd: np.ndarray, mask_et: np.ndarray):
    """
    x_etd: [E,T,D], mask_et: [E,T] (1=valid)
    Return:
      X_nd: [N,D] valid points only
      idx_n2: [N,2] where each row is (e,t) index for reconstruction
    """
    x = np.asarray(x_etd, dtype=np.float32)
    mask = (np.asarray(mask_et) > 0.5)
    idx = np.argwhere(mask)  # [N,2], columns: e, t
    if idx.size == 0:
        return np.zeros((0, x.shape[-1]), dtype=np.float32), idx.astype(np.int64)
    X = x[mask]  # [N,D]
    return X.astype(np.float32), idx.astype(np.int64)


def labels_from_sparse(idx_n2: np.ndarray, sparse_label_t: np.ndarray):
    # label per point by its timestep segment
    t = idx_n2[:, 1]
    return sparse_label_t[t].astype(np.int64)


def embed_tsne(X: np.ndarray, dim: int, perplexity: float, seed: int):
    X = np.asarray(X, dtype=np.float32)
    N = X.shape[0]
    if N < 3:
        return np.full((N, dim), np.nan, dtype=np.float32)
    max_perp = max(5.0, (N - 1) / 3.0)
    perp = float(min(perplexity, max_perp))
    tsne = TSNE(
        n_components=int(dim),
        perplexity=perp,
        init="pca",
        learning_rate="auto",
        random_state=int(seed),
    )
    return tsne.fit_transform(X).astype(np.float32)


def embed_pca(X: np.ndarray, dim: int):
    X = np.asarray(X, dtype=np.float32)
    if X.shape[0] < 2:
        return np.full((X.shape[0], dim), np.nan, dtype=np.float32)
    return PCA(n_components=int(dim)).fit_transform(X).astype(np.float32)


def pca_timeseries_mean_std(x_etd: np.ndarray, mask_et: np.ndarray):
    """
    Compute a PC1 curve over timesteps:
      1) Fit PC1 on all valid points (E*T points), using only x features.
      2) Project each point to PC1, reshape back to [E,T].
      3) mean±std across episodes at each timestep.
    """
    x = np.asarray(x_etd, dtype=np.float32)
    mask = (np.asarray(mask_et) > 0.5).reshape(-1)
    E, T, D = x.shape

    X = x.reshape(-1, D)
    Xv = X[mask]
    if Xv.shape[0] < 2:
        return (np.full((T,), np.nan, dtype=np.float32),
                np.full((T,), np.nan, dtype=np.float32))

    mean_vec = Xv.mean(axis=0, keepdims=True)
    Xc = Xv - mean_vec
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    pc1 = vt[0] / (np.linalg.norm(vt[0]) + 1e-8)

    y_flat = np.full((E * T,), np.nan, dtype=np.float32)
    y_all = (X - mean_vec.squeeze(0)) @ pc1
    y_flat[mask] = y_all[mask].astype(np.float32)
    y_et = y_flat.reshape(E, T)

    mean_t = np.nanmean(y_et, axis=0).astype(np.float32)
    std_t = np.nanstd(y_et, axis=0).astype(np.float32)
    return mean_t, std_t


def plot_mean_std_timeseries(out_path: Path, mean_t: np.ndarray, std_t: np.ndarray,
                             sparse_label: np.ndarray, title: str, y_label: str, dpi: int):
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(mean_t.shape[0])
    ax.plot(x, mean_t, linewidth=2)
    if np.isfinite(std_t).any() and np.nanmax(std_t) > 0:
        ax.fill_between(x, mean_t - std_t, mean_t + std_t, alpha=0.25)
    shade_sparse(ax, sparse_label)
    ax.set_xlabel("timestep")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_2d_scatter_points(out_path: Path, emb_n2: np.ndarray, lbl_n: np.ndarray,
                           title: str, dpi: int):
    lbl_n = np.asarray(lbl_n).astype(np.int64).reshape(-1)
    dense = (lbl_n == 0)
    sparse = (lbl_n == 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(emb_n2[dense, 0], emb_n2[dense, 1], s=10, alpha=0.8, label="dense")
    ax.scatter(emb_n2[sparse, 0], emb_n2[sparse, 1], s=10, alpha=0.8, label="sparse")

    ax.set_title(title)
    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def embed_1d_timeseries(x_etd: np.ndarray, mask_et: np.ndarray,
                        method: str, tsne_perplexity: float, tsne_seed: int):
    """
    Return mean_t, std_t computed from a 1D embedding of all valid points.
    """
    E, T, _ = x_etd.shape
    if method == "pca":
        return pca_timeseries_mean_std(x_etd, mask_et)

    # t-SNE: embed all valid points (no timestep feature), then aggregate across episodes per timestep
    X, idx = flatten_points(x_etd, mask_et)
    if X.shape[0] < 3:
        return (np.full((T,), np.nan, dtype=np.float32),
                np.full((T,), np.nan, dtype=np.float32))

    emb_n1 = embed_tsne(X, dim=1, perplexity=tsne_perplexity, seed=tsne_seed).reshape(-1)
    y_et = np.full((E, T), np.nan, dtype=np.float32)
    y_et[idx[:, 0], idx[:, 1]] = emb_n1.astype(np.float32)

    mean_t = np.nanmean(y_et, axis=0).astype(np.float32)
    std_t = np.nanstd(y_et, axis=0).astype(np.float32)
    return mean_t, std_t


def embed_2d_points(x_etd: np.ndarray, mask_et: np.ndarray, sparse_label_t: np.ndarray,
                    method: str, tsne_perplexity: float, tsne_seed: int):
    """
    Return (emb_n2, lbl_n) where emb is 2D embedding of all valid points,
    lbl is dense/sparse label per point (by its timestep).
    """
    X, idx = flatten_points(x_etd, mask_et)
    if X.shape[0] < 3:
        return np.full((X.shape[0], 2), np.nan, dtype=np.float32), np.zeros((X.shape[0],), dtype=np.int64)
    lbl = labels_from_sparse(idx, sparse_label_t)
    emb = embed_pca(X, 2) if method == "pca" else embed_tsne(X, 2, tsne_perplexity, tsne_seed)
    return emb.astype(np.float32), lbl.astype(np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--npz_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=300)

    parser.add_argument("--latent_dim", type=int, default=1, choices=[1, 2])
    parser.add_argument("--latent_method", type=str, default="pca", choices=["pca", "tsne"])

    parser.add_argument("--action_dim", type=int, default=1, choices=[1, 2])
    parser.add_argument("--action_method", type=str, default="raw", choices=["raw", "pca", "tsne"])
    parser.add_argument("--action_mode", type=str, default="norm", choices=["norm", "first"])

    parser.add_argument("--tsne_perplexity", type=float, default=20.0)
    parser.add_argument("--tsne_seed", type=int, default=0)
    args = parser.parse_args()

    if (args.log_dir is None) == (args.npz_path is None):
        raise ValueError("Provide exactly one of --log_dir or --npz_path.")

    npz_path = find_latest_npz(Path(args.log_dir)) if args.log_dir else Path(args.npz_path)
    out_dir = (npz_path.parent if args.out_dir is None else Path(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)

    mask = data["mask"].astype(np.float32)
    E, T = mask.shape

    if "sparse_label" in data:
        sparse_label = data["sparse_label"].astype(np.int64).reshape(-1)[:T]
    elif "sparse_segments" in data:
        sparse_label = make_sparse_label(T, data["sparse_segments"])
    else:
        sparse_label = np.zeros((T,), dtype=np.int64)

    # -------- 1) latent --------
    latent_mu = data["latent_mu"].astype(np.float32)

    if args.latent_dim == 1:
        mean_t, std_t = embed_1d_timeseries(
            latent_mu, mask,
            method=args.latent_method,
            tsne_perplexity=args.tsne_perplexity,
            tsne_seed=args.tsne_seed,
        )
        title = f"Latent trajectory ({args.latent_method.upper()}1): mean ± std over last episodes"
        plot_mean_std_timeseries(
            out_dir / "fig1_latent_1d.png",
            mean_t, std_t, sparse_label,
            title,
            "latent (1D)", args.dpi
        )
    else:
        emb_n2, lbl_n = embed_2d_points(
            latent_mu, mask, sparse_label,
            method=args.latent_method,
            tsne_perplexity=args.tsne_perplexity,
            tsne_seed=args.tsne_seed,
        )
        title = f"Latent embedding ({args.latent_method.upper()}2 on all points): dense vs sparse"
        plot_2d_scatter_points(out_dir / "fig1_latent_2d.png", emb_n2, lbl_n, title, args.dpi)

    # -------- 2) q --------
    # Priority:
    #   - q_prob: preferred (already probability)
    #   - q_aux2: in current run_bhera_vis.py this corresponds to q from q_head (also probability)
    #   - q_pred: fallback (might be derived)
    q_key = next((k for k in ["q_prob", "q_aux2", "q_pred"] if k in data), None)
    if q_key is None:
        raise KeyError("Expected one of q_prob / q_aux2 / q_pred in npz.")
    q = to_prob(data[q_key].astype(np.float32))
    q[mask <= 0.5] = np.nan
    q_mean = np.nanmean(q, axis=0).astype(np.float32)
    q_std = np.nanstd(q, axis=0).astype(np.float32)
    f1 = f1_at_threshold(q, sparse_label, mask, thr=args.thr)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(T)
    ax.plot(x, q_mean, linewidth=2, label="q_mean")
    ax.fill_between(x, q_mean - q_std, q_mean + q_std, alpha=0.25, label="q ± std")
    ax.plot(x, sparse_label.astype(np.float32), linewidth=2, label="GT sparse")
    shade_sparse(ax, sparse_label)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("timestep")
    ax.set_ylabel("q (0~1)")
    ax.set_title(f"q(sparse): mean ± std   (F1@{args.thr:.2f}={f1:.3f})   source={q_key}")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "fig2_q_traj.png", dpi=args.dpi)
    plt.close(fig)

    # -------- 3) action --------
    if args.action_method == "raw":
        key = "action_1d_norm" if args.action_mode == "norm" else "action_1d_first"
        a = data[key].astype(np.float32)
        a[mask <= 0.5] = np.nan
        a_mean = np.nanmean(a, axis=0).astype(np.float32)
        a_std = np.nanstd(a, axis=0).astype(np.float32)
        name = "||a||" if args.action_mode == "norm" else "a[0]"
        plot_mean_std_timeseries(
            out_dir / "fig3_action_1d.png",
            a_mean, a_std, sparse_label,
            f"Action trajectory (raw {name}): mean ± std over last episodes",
            f"action (1D: {name})", args.dpi
        )
    else:
        action_pool = data["action_pool"].astype(np.float32)  # [E,T,A]
        if args.action_dim == 1:
            mean_t, std_t = embed_1d_timeseries(
                action_pool, mask,
                method=args.action_method,
                tsne_perplexity=args.tsne_perplexity,
                tsne_seed=args.tsne_seed,
            )
            title = f"Action trajectory ({args.action_method.upper()}1): mean ± std over last episodes"
            plot_mean_std_timeseries(
                out_dir / "fig3_action_1d.png",
                mean_t, std_t, sparse_label,
                title,
                "action (1D)", args.dpi
            )
        else:
            emb_n2, lbl_n = embed_2d_points(
                action_pool, mask, sparse_label,
                method=args.action_method,
                tsne_perplexity=args.tsne_perplexity,
                tsne_seed=args.tsne_seed,
            )
            title = f"Action embedding ({args.action_method.upper()}2 on all points): dense vs sparse"
            plot_2d_scatter_points(out_dir / "fig3_action_2d.png", emb_n2, lbl_n, title, args.dpi)

    print(f"[OK] Loaded: {npz_path}")
    print(f"[OK] Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
