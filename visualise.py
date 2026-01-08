#!/usr/bin/env python3
"""
Visualize guided diffusion samples in the same feature/PCA space as the reference guiding data.

The embedding pipeline mirrors the analysis performed when generating the transition-state summary:
1. Build a feature matrix per frame consisting of all CA–CA pairwise distances (Å) and
   sine/cosine encodings of backbone φ/ψ torsion angles computed from local frames.
2. Standardize the feature space and fit a 2D PCA model on the reference (guiding) trajectories.
3. Project both the reference data and newly generated samples into that PCA space for comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PN_VECTOR = torch.tensor((-0.526, 1.363, 0.0), dtype=torch.float32)
PC_VECTOR = torch.tensor((1.526, 0.0, 0.0), dtype=torch.float32)


def compute_dihedral(p1, p2, p3, p4, eps: float = 1e-8):
    """Signed dihedral (radians) between planes defined by consecutive triplets."""

    b2 = p3 - p2
    b2_norm = b2 / (b2.norm(dim=-1, keepdim=True) + eps)

    v0 = p1 - p2
    v1 = p4 - p3

    v0p = v0 - (v0 * b2_norm).sum(dim=-1, keepdim=True) * b2_norm
    v1p = v1 - (v1 * b2_norm).sum(dim=-1, keepdim=True) * b2_norm

    x = (v0p * v1p).sum(dim=-1)
    y = (torch.cross(b2_norm, v0p, dim=-1) * v1p).sum(dim=-1)
    return torch.atan2(y, x)


def compute_backbone_torsions(
    positions: np.ndarray,
    orientations: np.ndarray | None,
    *,
    device: torch.device,
    pN: torch.Tensor = PN_VECTOR,
    pC: torch.Tensor = PC_VECTOR,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute backbone φ/ψ torsion angles using CA positions and local frames."""

    if orientations is None:
        return np.empty((positions.shape[0], 0)), np.empty((positions.shape[0], 0))

    ca = torch.as_tensor(positions, dtype=torch.float32, device=device)
    Q = torch.as_tensor(orientations, dtype=torch.float32, device=device)

    N = ca + torch.einsum("...ij,j->...i", Q, pN.to(device))
    C = ca + torch.einsum("...ij,j->...i", Q, pC.to(device))
    C_prev = torch.roll(C, shifts=1, dims=1)
    N_next = torch.roll(N, shifts=-1, dims=1)

    phi = compute_dihedral(C_prev, N, ca, C)
    psi = compute_dihedral(N, ca, C, N_next)

    phi = phi[:, 1:]
    psi = psi[:, :-1]
    return phi.detach().cpu().numpy(), psi.detach().cpu().numpy()


def torsion_feature_matrix(
    positions: np.ndarray,
    orientations: np.ndarray | None,
    *,
    device: torch.device,
) -> np.ndarray:
    """Return sine/cosine encodings of backbone torsion angles."""
    phi, psi = compute_backbone_torsions(positions, orientations, device=device)
    blocks: list[np.ndarray] = []
    if phi.size:
        blocks.extend([np.sin(phi), np.cos(phi)])
    if psi.size:
        blocks.extend([np.sin(psi), np.cos(psi)])
    if not blocks:
        return np.empty((positions.shape[0], 0))
    return np.hstack(blocks)


def pairwise_distance_features(
    positions: np.ndarray,
    pair_indices: torch.Tensor | np.ndarray | None = None,
    *,
    scale_to_angstrom: bool = True,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    """Flattened CA–CA pairwise distances for every frame using the requested device."""

    tensor = torch.as_tensor(positions, dtype=torch.float32, device=device)
    batch, n_res, _ = tensor.shape

    if pair_indices is None:
        pair_indices = torch.triu_indices(n_res, n_res, offset=1, device=device)
    elif isinstance(pair_indices, np.ndarray):
        pair_indices = torch.as_tensor(pair_indices.T, dtype=torch.long, device=device)
    else:
        pair_indices = pair_indices.to(device)

    diffs = tensor[:, pair_indices[0], :] - tensor[:, pair_indices[1], :]
    distances = torch.linalg.norm(diffs, dim=-1)
    if scale_to_angstrom:
        distances = distances * 10.0
    return distances.reshape(batch, -1).detach().cpu().numpy(), pair_indices.detach().cpu()


def build_feature_matrix(
    positions: np.ndarray,
    orientations: np.ndarray | None,
    pair_indices: torch.Tensor | np.ndarray,
    *,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    """Concatenate distance and torsion-based features with consistent ordering."""

    distances, pair_indices = pairwise_distance_features(
        positions, pair_indices=pair_indices, device=device
    )
    torsions = torsion_feature_matrix(positions, orientations, device=device)
    features = [distances]
    if torsions.size:
        features.append(torsions)
    return np.hstack(features), pair_indices


@dataclass
class EmbeddingResult:
    reference_pc: np.ndarray
    sample_pc: np.ndarray
    explained_variance_ratio: np.ndarray


def project_samples(
    reference_positions: np.ndarray,
    reference_orientations: np.ndarray | None,
    sample_positions: np.ndarray,
    sample_orientations: np.ndarray | None,
    pair_indices: np.ndarray,
    *,
    device: torch.device,
    n_components: int = 2,
) -> EmbeddingResult:
    """Fit scaler+PCA on reference features and apply them to both reference and new samples."""

    ref_features, pair_indices = build_feature_matrix(
        reference_positions, reference_orientations, pair_indices, device=device
    )
    scaler = StandardScaler()
    # apparently we should not use this
    # ref_features_scaled = scaler.fit_transform(ref_features)

    pca = PCA(n_components=n_components)
    # ref_pc = pca.fit_transform(ref_features_scaled)
    ref_pc = pca.fit_transform(ref_features)

    sample_features, _ = build_feature_matrix(
        sample_positions, sample_orientations, pair_indices, device=device
    )
    sample_pc = pca.transform(sample_features)

    return EmbeddingResult(
        reference_pc=ref_pc,
        sample_pc=sample_pc,
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def plot_embedding(
    embedding: EmbeddingResult,
    macro_assignments: np.ndarray | None,
    macro_labels: Iterable[str] | None,
    output_path: Path | None,
    title: str | None,
):
    """Scatter plot of reference vs. guided samples in PCA space."""

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    if macro_assignments is not None:
        unique = np.unique(macro_assignments)
        labels = list(macro_labels) if macro_labels is not None else [f"macro {i}" for i in unique]
        for macro in unique:
            mask = macro_assignments == macro
            # print percentage of frames in this macrostate
            print(f"Percentage of frames in macrostate {macro}: {np.sum(mask) / len(macro_assignments) * 100:.2f}%")
            label = labels[int(macro)] if int(macro) < len(labels) else f"macro {macro}"
            ax.scatter(
                embedding.reference_pc[mask, 0],
                embedding.reference_pc[mask, 1],
                s=12,
                alpha=0.35,
                edgecolors="none",
                color=colors[int(macro) % len(colors)],
                label=label,
            )
    else:
        ax.scatter(
            embedding.reference_pc[:, 0],
            embedding.reference_pc[:, 1],
            s=12,
            alpha=0.35,
            color="tab:blue",
            edgecolors="none",
            label="reference frames",
        )

    ax.scatter(
        embedding.sample_pc[:, 0],
        embedding.sample_pc[:, 1],
        s=80,
        marker="*",
        color="black",
        edgecolors="white",
        linewidths=0.5,
        label="guided samples",
    )

    ax.set_xlabel(f"PC1 ({embedding.explained_variance_ratio[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({embedding.explained_variance_ratio[1]*100:.1f}% var)")
    # ax.set_xlim(-10, 50)
    ax.set_title(title or "Guided samples vs. reference PCA")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2, linewidth=0.5)

    if output_path is not None:
        # output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved PCA scatter to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def _load_npz(path: Path, pos_key: str, rot_key: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    print(path)
    with np.load(path, allow_pickle=True) as data:
        print(data.keys())
        positions = data[pos_key]
        orientations = data.get(rot_key) if rot_key else None
    return positions, orientations


def _reshape_positions(array: np.ndarray, n_residues: int) -> np.ndarray:
    """Ensure sample positions are shaped (batch, n_residues, 3)."""

    if array.ndim == 3:
        if array.shape[1:] != (n_residues, 3):
            raise ValueError(
                f"Expected positions with shape (batch, {n_residues}, 3); got {array.shape}."
            )
        return array
    if array.ndim == 2 and array.shape[1] == 3:
        if array.shape[0] % n_residues != 0:
            raise ValueError(
                f"Cannot reshape positions of shape {array.shape} into batches of length {n_residues}."
            )
        batch = array.shape[0] // n_residues
        return array.reshape(batch, n_residues, 3)
    raise ValueError(
        f"Positions array must have ndim 2 or 3 with trailing dimension 3; got {array.shape}."
    )


def _reshape_orientations(array: np.ndarray | None, n_residues: int) -> np.ndarray | None:
    """Ensure sample orientations are shaped (batch, n_residues, 3, 3)."""

    if array is None:
        return None
    if array.ndim == 4:
        if array.shape[1:] != (n_residues, 3, 3):
            raise ValueError(
                f"Expected orientations with shape (batch, {n_residues}, 3, 3); got {array.shape}."
            )
        return array
    if array.ndim == 3 and array.shape[1:] == (3, 3):
        if array.shape[0] % n_residues != 0:
            raise ValueError(
                f"Cannot reshape orientations of shape {array.shape} into batches of length {n_residues}."
            )
        batch = array.shape[0] // n_residues
        return array.reshape(batch, n_residues, 3, 3)
    raise ValueError(
        f"Orientations array must have ndim 3 or 4 with trailing (3, 3); got {array.shape}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize guided samples in the PCA space defined by reference trajectories."
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the guiding/transition-state NPZ (e.g., state_analysis_summary.npz).",
    )
    parser.add_argument(
        "--samples",
        required=True,
        help="Path to samples.npz produced by path_guidance.py.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=None,
        help="Optional output path for the PCA scatter plot (PNG).",
    )
    parser.add_argument(
        "--components", type=int, default=2, help="Number of PCA components to retain."
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional custom title for the plot.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Computation device for feature extraction (default: cuda if available else cpu).",
    )
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but no GPU is available.")
        device = torch.device(args.device)

    ref_path = Path(args.reference)
    sample_path = Path(args.samples)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference file not found: {ref_path}")
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    ref_positions, ref_orientations = _load_npz(ref_path, pos_key="ca_positions", rot_key="node_orientations")
    sample_positions, sample_orientations = _load_npz(sample_path, pos_key="pos", rot_key="node_orientations")

    sample_positions = _reshape_positions(sample_positions, ref_positions.shape[1])
    sample_orientations = _reshape_orientations(sample_orientations, ref_positions.shape[1])
    _, pair_indices = pairwise_distance_features(
        ref_positions, pair_indices=None, device=device
    )
    embedding = project_samples(
        reference_positions=ref_positions,
        reference_orientations=ref_orientations,
        sample_positions=sample_positions,
        sample_orientations=sample_orientations,
        pair_indices=pair_indices,
        device=device,
        n_components=args.components,
    )

    macros = None
    macro_labels = None
    with np.load(ref_path, allow_pickle=True) as ref_npz:
        macros = ref_npz.get("macrostate_assignment_per_frame")
        macro_labels = ref_npz.get("macrostate_labels")

    if args.save_path is None:
        save_path = sample_path.parent
    else:
        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path / "visualisation.png"

    plot_embedding(
        embedding=embedding,
        macro_assignments=macros,
        macro_labels=macro_labels,
        output_path=save_path,
        title=args.title,
    )

    explained = embedding.explained_variance_ratio
    print(
        f"Explained variance ratio: "
        + ", ".join(f"PC{i+1}={ratio*100:.2f}%" for i, ratio in enumerate(explained))
    )
    print(f"Projected {embedding.sample_pc.shape[0]} samples into PCA space.")


if __name__ == "__main__":
    main()

