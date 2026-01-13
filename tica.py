import numpy as np
from deeptime.decomposition import TICA
from sklearn.cluster import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov import pcca
from deeptime.markov.tools.analysis import committor
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import argparse


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
        print("orientations are None")
        return np.empty((positions.shape[0], 0)), np.empty((positions.shape[0], 0))

    ca = torch.as_tensor(positions * 10.0, dtype=torch.float32, device=device)
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


def build_feature_matrix(
    positions: np.ndarray,
    orientations: np.ndarray | None,
    *,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    """Concatenate distance and torsion-based features with consistent ordering."""

    bond_vectors = positions[:, 1:, :] - positions[:, :-1, :]
    consecutive_ca_distances = np.linalg.norm(bond_vectors, axis=-1)
    distances = consecutive_ca_distances * 10.0 

    distance_cutoff = 6.0  # Å
    valid_frame_mask = np.all(distances <= distance_cutoff, axis=1)
    num_removed = np.count_nonzero(~valid_frame_mask)

    if num_removed > 0:
        positions = positions[valid_frame_mask]
        orientations = orientations[valid_frame_mask]
        distances = distances[valid_frame_mask]

    torsions = torsion_feature_matrix(positions, orientations, device=device)
    return torsions


def _load_npz(path: Path, pos_key: str, rot_key: str | None) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=True) as data:
        positions = data[pos_key]
        orientations = data.get(rot_key) if rot_key else None
    return positions, orientations


def save_figure(figure, save_path: Path):
    figure.savefig(save_path)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize guided samples in the PCA space defined by reference trajectories."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Path to the guiding/transition-state NPZ (e.g., state_analysis_summary.npz).",
    )
    parser.add_argument(
        "--samples",
        type=Path,
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
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Computation device for feature extraction (default: cuda if available else cpu).",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="Maximum number of files to load.",
    )
    args = parser.parse_args()

    RANDOM_SEED = 42
    rng = np.random.RandomState(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    feat_ref = np.load(args.reference)["feat_ref"]

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but no GPU is available.")
        device = torch.device(args.device)

    sample_path = Path(args.samples)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    # check if the ref_path is a directory
    if sample_path.is_dir():
        data_files = list(sample_path.glob("*.npz"))
        if args.max_files is not None:
            data_files = data_files[:args.max_files]
        positions = []
        orientations = []
        for file in data_files:
            sample_positions, sample_orientations = _load_npz(file, pos_key="pos", rot_key="node_orientations")
            positions.append(sample_positions)
            orientations.append(sample_orientations)
        sample_positions = np.concatenate(positions, axis=0)
        sample_orientations = np.concatenate(orientations, axis=0)
    else:
        sample_positions, sample_orientations = _load_npz(sample_path, pos_key="pos", rot_key="node_orientations")

    if sample_positions.ndim == 2:
        sample_positions = sample_positions.reshape(-1, 56, 3)
    if sample_orientations.ndim == 3:
        sample_orientations = sample_orientations.reshape(-1, 56, 3, 3)

    print(sample_positions.shape, sample_orientations.shape)

    feat_sample = build_feature_matrix(sample_positions, sample_orientations, device=device)

    tica = TICA(dim=2, lagtime=100).fit(feat_ref)
    print(feat_ref.shape, feat_sample.shape)
    tica_ref = tica.transform(feat_ref)
    tica_sample = tica.transform(feat_sample)


    N_CLUSTERS = 100
    LAG_TIME_MSM = 100
    N_MACROSTATES = 6
    TOP_N_CLUSTERS = 10
    COMMITTOR_SOURCE_STATE = 0
    COMMITTOR_TARGET_STATE = 1
    TRANSITION_THRESHOLD_LOW = 0.4
    TRANSITION_THRESHOLD_HIGH = 0.6

    np.random.seed(RANDOM_SEED)
    embedding_for_clustering = tica_ref
    clustering = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
    clustering.fit(embedding_for_clustering)
    dtrajs = clustering.labels_

    counts_estimator = TransitionCountEstimator(lagtime=LAG_TIME_MSM, count_mode="sliding")
    counts = counts_estimator.fit(dtrajs).fetch_model()
    msm_estimator = MaximumLikelihoodMSM(reversible=True, stationary_distribution_constraint=None)
    msm = msm_estimator.fit(counts).fetch_model()

    np.random.seed(RANDOM_SEED)
    pcca_obj = pcca(msm.transition_matrix, N_MACROSTATES)

    state_clusters = []
    for i in range(N_MACROSTATES):
        clusters = np.argsort(pcca_obj.memberships[:, i])[-TOP_N_CLUSTERS:]
        state_clusters.append(clusters)

    macrostate_assignments = np.argmax(pcca_obj.memberships, axis=1)

    source_clusters = state_clusters[COMMITTOR_SOURCE_STATE]
    target_clusters = state_clusters[COMMITTOR_TARGET_STATE]
    target_clusters = np.setdiff1d(target_clusters, source_clusters)

    committor_probs = committor(msm.transition_matrix, source_clusters, target_clusters)

    transition_microstates = np.where(
        (committor_probs >= TRANSITION_THRESHOLD_LOW) & 
        (committor_probs <= TRANSITION_THRESHOLD_HIGH)
    )[0]

    if args.save_path is None:
        if args.samples.is_dir():
            save_path = args.samples
        else:
            save_path = args.samples.parent
    else:
        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

    plt.hist(
        committor_probs,
        weights=msm.stationary_distribution,
        bins=50,
        density=True,
        histtype="step",
        linewidth=2,
        label="Stationary distribution weighted",
        color='blue'
    )
    plt.axvline(TRANSITION_THRESHOLD_LOW, color="r", linestyle="--", alpha=0.7, label=f"Transition threshold ({TRANSITION_THRESHOLD_LOW})")
    plt.axvline(TRANSITION_THRESHOLD_HIGH, color="r", linestyle="--", alpha=0.7, label=f"Transition threshold ({TRANSITION_THRESHOLD_HIGH})")
    plt.xlabel("Committor probability")
    plt.ylabel("Weighted density")
    plt.title(f"Committor Probability Distribution\n(State {COMMITTOR_SOURCE_STATE} -> State {COMMITTOR_TARGET_STATE})")
    plt.legend(fontsize=8)
    plt.semilogy()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path / "committor_probability_distribution.png")
    plt.close()

    is_transition = np.isin(dtrajs, transition_microstates)
    dtrajs_sample = clustering.predict(tica_sample)
    is_transition_sample = np.isin(dtrajs_sample, transition_microstates)

    plt.scatter(embedding_for_clustering[~is_transition, 0], embedding_for_clustering[~is_transition, 1], s=1, alpha=0.3, c='lightgray', label='Reference (non-transition)')
    plt.scatter(embedding_for_clustering[is_transition, 0], embedding_for_clustering[is_transition, 1], c='blue', alpha=0.8, s=5, marker='x', label=f'Reference (transition, n={np.sum(is_transition)})')
    plt.scatter(tica_sample[~is_transition_sample, 0], tica_sample[~is_transition_sample, 1], s=1, alpha=0.3, c='black', label='XTC (non-transition)')
    plt.scatter(tica_sample[is_transition_sample, 0], tica_sample[is_transition_sample, 1], c='red', alpha=0.8, s=5, marker='x', label=f'XTC (transition, n={np.sum(is_transition_sample)})')
    plt.xlabel('tIC 1')
    plt.ylabel('tIC 2')
    plt.title('Transition States in TICA Space')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path / "transition_states_tica_space.png")
    plt.close()


if __name__ == "__main__":
    main()