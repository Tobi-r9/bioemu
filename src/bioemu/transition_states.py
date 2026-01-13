# transition_classifier.py
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


# ---- Feature construction (mirrors visualise.py / tmp_2.py) ----------------- #

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


class TransitionClassifier:

    def __init__(
        self,
        npz_path: str,
        device: str | torch.device | None = None,
    ):
        """
        Parameters
        ----------
        npz_path : str
            Path to feat_ref.npz for this protein.
        device : str or torch.device or None
            Compute device for feature extraction (None -> cuda if available).
        """
        self.npz_path = npz_path
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.feat_ref = np.load(self.npz_path)["feat_ref"]
        self._fit_tica()
        self._define_microstates()

    def _fit_tica(self):
        self.tica = TICA(dim=2, lagtime=100).fit(self.feat_ref)
        self.tica_ref = self.tica.transform(self.feat_ref)

    def _define_microstates(self):
        N_CLUSTERS = 100
        LAG_TIME_MSM = 100
        N_MACROSTATES = 6
        TOP_N_CLUSTERS = 10
        COMMITTOR_SOURCE_STATE = 0
        COMMITTOR_TARGET_STATE = 1
        TRANSITION_THRESHOLD_LOW = 0.4
        TRANSITION_THRESHOLD_HIGH = 0.6

        RANDOM_SEED = 42
        np.random.seed(RANDOM_SEED)
        embedding_for_clustering = self.tica_ref
        self.clustering = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_SEED, n_init=10)
        self.clustering.fit(embedding_for_clustering)
        dtrajs = self.clustering.labels_

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

        source_clusters = state_clusters[COMMITTOR_SOURCE_STATE]
        target_clusters = state_clusters[COMMITTOR_TARGET_STATE]
        target_clusters = np.setdiff1d(target_clusters, source_clusters)

        committor_probs = committor(msm.transition_matrix, source_clusters, target_clusters)

        self.transition_microstates = np.where(
            (committor_probs >= TRANSITION_THRESHOLD_LOW) & 
            (committor_probs <= TRANSITION_THRESHOLD_HIGH)
        )[0]


    def classify(
        self,
        new_ca_positions: np.ndarray,
        new_node_orientations: np.ndarray = None,
    ):

        feat_sample = build_feature_matrix(new_ca_positions, new_node_orientations, device=self.device)
        if feat_sample.shape[0] == 0:
            return {
                "assigned_microstates": np.zeros(new_ca_positions.shape[0]),
                "is_transition": np.zeros(new_ca_positions.shape[0]),
            }
        tica_sample = self.tica.transform(feat_sample)
        assigned_microstates = self.clustering.predict(tica_sample)
        is_transition_state = np.isin(assigned_microstates, self.transition_microstates)

        return {
            "assigned_microstates": assigned_microstates,
            "is_transition": is_transition_state,
        }


if __name__ == "__main__":

    def _load_npz(path: Path, pos_key: str, rot_key: str | None) -> tuple[np.ndarray, np.ndarray | None]:
        with np.load(path, allow_pickle=True) as data:
            positions = data[pos_key]
            orientations = data.get(rot_key) if rot_key else None
        return positions, orientations

    data_dir = Path("/lustre/groups/bauer/code/thoeppe/bioemu/protein_data")
    sample_dir = data_dir / "protein_heun_64_100000_sequence"
    data_files = list(sample_dir.glob("*.npz"))
    positions = []
    orientations = []
    for file in data_files[:10]:
        sample_positions, sample_orientations = _load_npz(file, pos_key="pos", rot_key="node_orientations")
        positions.append(sample_positions)
        orientations.append(sample_orientations)
    sample_positions = np.concatenate(positions, axis=0)
    sample_orientations = np.concatenate(orientations, axis=0)

    print(sample_positions.shape, sample_orientations.shape)

    clf = TransitionClassifier(
        npz_path=data_dir / "feats_ref" / "feat_ref.npz",
    )

    result = clf.classify(
        new_ca_positions=sample_positions,
        new_node_orientations=sample_orientations,
    )

    print("transition-like indices:", np.where(result["is_transition"])[0])
    print("assigned microstates:", result["assigned_microstates"])
    print("percentage of transition-like states:", np.mean(result["is_transition"]))
