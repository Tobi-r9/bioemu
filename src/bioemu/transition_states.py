# transition_classifier.py

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from deeptime.markov.tools.analysis import committor
# from bioemu.path_guidance import load_initial_structure


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


class TransitionClassifier:

    def __init__(
        self,
        npz_path: str,
        n_neighbors: int = 5,
        use_rotations: bool = True,
        n_components: int = 2,
        device: str | torch.device | None = None,
    ):
        """
        Parameters
        ----------
        npz_path : str
            Path to state_analysis_summary.npz for this protein.
        n_neighbors : int
            k for k-NN (majority vote over nearest MD frames).
        use_rotations : bool
            If True, include rotation matrices in the feature vector.
        n_components : int
            Number of PCA components to retain for the embedding used by k-NN.
        device : str or torch.device or None
            Compute device for feature extraction (None -> cuda if available).
        """
        self.npz_path = npz_path
        self.n_neighbors = n_neighbors
        self.use_rotations = use_rotations
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.n_components = n_components

        self._load_data()
        self._build_features()
        self._fit_knn()
        self._compute_committor_per_state()

    def _load_data(self):
        data = np.load(self.npz_path, allow_pickle=True)

        self.ca_positions = data["ca_positions"]               # (n_frames, n_res, 3)
        self.node_orientations = data["node_orientations"]     # (n_frames, n_res, 3, 3)
        self.discrete_states = data["discrete_states"]         # (n_frames,)
        self.T = data["transition_matrix"]                     # (n_states, n_states)
        self.macro_per_state = data["macrostate_assignment_per_state"]  # (n_states,)
        self.n_frames, self.n_residues, _ = self.ca_positions.shape
        self.n_states = self.T.shape[0]

    def _build_features(self):
        """Build reference feature matrix consistent with training pipeline."""
        orientations = self.node_orientations if self.use_rotations else None
        self.pair_indices = torch.triu_indices(
            self.n_residues, self.n_residues, offset=1, device=self.device
        )
        ref_features, self.pair_indices = build_feature_matrix(
            self.ca_positions,
            orientations,
            self.pair_indices,
            device=self.device,
        )
        self.ref_features = ref_features
        # Fit PCA on the same features used by the MSM/PCA workflow.
        self.pca = PCA(n_components=self.n_components)
        self.ref_pc = self.pca.fit_transform(ref_features)

    def _fit_knn(self):
        """
        Fit a k-NN model on MD frames in feature space to enable
        assignment of new samples to microstates.
        """
        self.knn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="auto",
            metric="euclidean",
        )
        self.knn.fit(self.ref_pc)

    def _compute_committor_per_state(self):
        """
        Compute committor probability q_i for each microstate i using
        the MSM transition matrix and macrostate assignments.
        States labeled 2 are "intermediate"/transition region.
        """
        A = np.where(self.macro_per_state == 0)[0]
        B = np.where(self.macro_per_state == 1)[0]

        if len(A) == 0 or len(B) == 0:
            raise ValueError("Need at least one state_0 and one state_1 microstate to compute committor.")

        self.q_state = committor(self.T, A, B)  # shape (n_states,)


    def _features_from_new_samples(self, new_ca, new_rot):
        """Build PCA-projected feature vectors for new samples."""
        if new_ca.shape[1] != self.n_residues:
            raise ValueError(
                f"New samples have seq_len={new_ca.shape[1]}, "
                f"but MSM model has n_residues={self.n_residues}."
            )

        orientations = None
        if self.use_rotations:
            if new_rot is None:
                raise ValueError("Rotation matrices must be provided when use_rotations=True.")
            orientations = new_rot

        features, _ = build_feature_matrix(
            new_ca,
            orientations,
            self.pair_indices,
            device=self.device,
        )
        return self.pca.transform(features)

    def classify(
        self,
        new_ca_positions: np.ndarray,
        new_node_orientations: np.ndarray = None,
        q_transition_low: float = 0.2,
        q_transition_high: float = 0.8,
    ):
        """
        Classify new samples as transition / non-transition.

        Parameters
        ----------
        new_ca_positions : (b, n_res, 3) array
        new_node_orientations : (b, n_res, 3, 3) array or None
        q_transition_low, q_transition_high : float
            Committor thresholds to call a state "transition-like".

        Returns
        -------
        result : dict
            {
                "assigned_microstates": (b,) int,
                "committor": (b,) float,
                "is_transition": (b,) bool,
                "macrostate_from_state": (b,) int,  # 0,1,2 from macro_per_state
            }
        """
        features_std = self._features_from_new_samples(new_ca_positions, new_node_orientations)

        # Find nearest MD frames in feature space
        distances, indices = self.knn.kneighbors(features_std, return_distance=True)

        # Majority vote over their microstate labels
        neighbor_states = self.discrete_states[indices]  # (b, k)
        assigned_microstates = np.zeros(new_ca_positions.shape[0], dtype=int)
        for i in range(new_ca_positions.shape[0]):
            vals, counts = np.unique(neighbor_states[i], return_counts=True)
            assigned_microstates[i] = vals[np.argmax(counts)]

        # Committor & macrostate
        q_state = self.q_state[assigned_microstates]
        macro = self.macro_per_state[assigned_microstates]

        is_transition_state = (q_state > q_transition_low) & (q_state < q_transition_high)

        return {
            "assigned_microstates": assigned_microstates,
            "committor": q_state,
            "is_transition": is_transition_state,
            "macrostate_from_state": macro,
        }


if __name__ == "__main__":

    data_dir = "/lustre/groups/bauer/code/thoeppe/bioemu/protein_data/transition_states"
    init_npz = data_dir + "/state_1_positions_orientations.npz"
    sequence = data_dir + "/msa.a3m"
    bio_ca, bio_rot, sequence = load_initial_structure(init_npz, sequence)
    print(bio_ca.shape, bio_rot.shape, sequence)

    clf = TransitionClassifier(
        npz_path=data_dir + "/state_analysis_summary.npz",
        n_neighbors=1,
        use_rotations=True,
    )

    result = clf.classify(
        new_ca_positions=bio_ca,
        new_node_orientations=bio_rot,
        q_transition_low=0.3,
        q_transition_high=0.7,
    )

    print("committor:", result["committor"])
    print("transition-like indices:", np.where(result["is_transition"])[0])
    print("macrostate labels:", result["macrostate_from_state"])
    print("percentage of transition-like states:", np.mean(result["is_transition"]))
