# transition_classifier.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
from deeptime.markov.tools.analysis import committor
# from bioemu.path_guidance import load_initial_structure


class TransitionClassifier:

    def __init__(
        self,
        npz_path: str,
        n_neighbors: int = 5,
        use_rotations: bool = True,
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
        """
        self.npz_path = npz_path
        self.n_neighbors = n_neighbors
        self.use_rotations = use_rotations

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
        pos_flat = self.ca_positions.reshape(self.n_frames, -1)

        # TODO: Should rotations be used? This was not the case in the tito project.    
        if self.use_rotations:
            rot_flat = self.node_orientations.reshape(self.n_frames, -1)
            features = np.concatenate([pos_flat, rot_flat], axis=1)
        else:
            features = pos_flat

        # TODO: should we standardize the features?
        self.feature_mean = features.mean(axis=0, keepdims=True)
        self.feature_std = features.std(axis=0, keepdims=True)
        self.feature_std[self.feature_std == 0.0] = 1.0

        self.ref_features = (features - self.feature_mean) / self.feature_std

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
        self.knn.fit(self.ref_features)

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
        """
        Build standardized feature vectors for new samples.

        Parameters
        ----------
        new_ca : (b, n_res, 3)
        new_rot : (b, n_res, 3, 3) or None

        Returns
        -------
        features_std : (b, d)
        """
        if new_ca.shape[1] != self.n_residues:
            raise ValueError(
                f"New samples have seq_len={new_ca.shape[1]}, "
                f"but MSM model has n_residues={self.n_residues}."
            )

        pos_flat = new_ca.reshape(new_ca.shape[0], -1)

        if self.use_rotations:
            if new_rot is None:
                raise ValueError("Rotation matrices must be provided when use_rotations=True.")
            rot_flat = new_rot.reshape(new_rot.shape[0], -1)
            features = np.concatenate([pos_flat, rot_flat], axis=1)
        else:
            features = pos_flat

        features_std = (features - self.feature_mean) / self.feature_std
        return features_std

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
