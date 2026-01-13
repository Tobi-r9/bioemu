from pathlib import Path

import numpy as np
import torch

from .denoiser import forward_probability_flow_trajectory
from .model_utils import load_model, load_sdes, maybe_download_checkpoint
from .sample_utils import build_chemgraph, reshape_positions, reshape_orientations, guided_reverse_integration, load_initial_structure
from .transition_states import TransitionClassifier

import os
# torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  

def _ensure_frames_tensor(positions: torch.Tensor) -> torch.Tensor:
    """
    Ensure positions are shaped as [F, N, 3] where F aggregates any leading dimensions.
    Accepts [N,3], [F,N,3], or higher rank [..., N, 3].
    """
    if positions.dim() == 2:
        return positions.unsqueeze(0)
    if positions.dim() > 3:
        leading = int(np.prod(positions.shape[:-2]))
        return positions.reshape(leading, positions.shape[-2], positions.shape[-1])
    return positions


def _wasserstein_1d_uniform_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    Fallback 1D Wasserstein distance for uniformly weighted samples without SciPy.
    Computes ∫|F_a(x) - F_b(x)| dx using a merge-walk over sorted samples.
    """
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0:
        return float(np.mean(np.abs(b - b.min())))
    if b.size == 0:
        return float(np.mean(np.abs(a - a.min())))
    a = np.sort(a)
    b = np.sort(b)
    i = j = 0
    wa = 1.0 / a.size
    wb = 1.0 / b.size
    cdfa = cdfb = 0.0
    last_x = min(a[0], b[0])
    total = 0.0
    inf = float("inf")
    while i < a.size or j < b.size:
        next_a = a[i] if i < a.size else inf
        next_b = b[j] if j < b.size else inf
        x = next_a if next_a <= next_b else next_b
        total += abs(cdfa - cdfb) * (x - last_x)
        if next_a <= next_b:
            i += 1
            cdfa += wa
        if next_b <= next_a:
            j += 1
            cdfb += wb
        last_x = x
    return float(total)


def compute_mean_wasserstein_bond_distance(
    edge_index: torch.Tensor,
    positions_a: torch.Tensor,
    positions_b: torch.Tensor,
    num_edges: int | None = None,
) -> float:
    """
    Compute the mean 1D Wasserstein distance between bond length distributions of two
    position sets (A vs B). For each bond (i,j), we form distributions of distances
    across frames/batch and compute W1 distance; the final metric is the mean over bonds.
    Args:
        edge_index: Tensor of shape [2, E] with bond indices.
        positions_a: Tensor shaped [F_a,N,3] or [N,3] or higher rank [...,N,3].
        positions_b: Tensor shaped [F_b,N,3] or [N,3] or higher rank [...,N,3].
        num_edges: If provided, limit to the first num_edges bonds (for speed).
    Returns:
        Mean Wasserstein distance over the selected bonds (float).
    """
    edge_index = edge_index.detach().cpu().long()
    if num_edges is not None:
        num_edges = int(num_edges)
        edge_index = edge_index[:, : min(num_edges, edge_index.shape[1])]
    positions_a = _ensure_frames_tensor(positions_a.detach().cpu())
    positions_b = _ensure_frames_tensor(positions_b.detach().cpu())

    try:
        from scipy.stats import wasserstein_distance as _scipy_wd  # type: ignore
        def wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
            return float(_scipy_wd(u, v))
    except Exception:
        def wasserstein_1d(u: np.ndarray, v: np.ndarray) -> float:
            return _wasserstein_1d_uniform_np(u, v)

    distances: list[float] = []
    for i, j in edge_index.t():
        i_idx = int(i.item())
        j_idx = int(j.item())
        d_a = torch.linalg.norm(positions_a[:, i_idx, :] - positions_a[:, j_idx, :], dim=-1).numpy()
        d_b = torch.linalg.norm(positions_b[:, i_idx, :] - positions_b[:, j_idx, :], dim=-1).numpy()
        d_a = d_a[np.isfinite(d_a)]
        d_b = d_b[np.isfinite(d_b)]
        wd = wasserstein_1d(d_a, d_b)
        distances.append(wd)

    if len(distances) == 0:
        return 0.0
    return float(np.mean(distances))


@torch.no_grad()
def main(
    sequence: str | Path,
    init_npz: str | Path,
    batch_size: int = 64,
    N: int = 200,
    results_dir: str | Path = "tmp_path_guidance_samples",
    eps_t: float = 1e-3,
    max_t: float = 1.0,
    method: str = "heun",
    model_name: str | None = "bioemu-v1.1",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
) -> None:
    """
    Deterministic probability-flow roundtrip check using the trained score model:
    - Forward PF-ODE from eps_t -> max_t
    - Reverse PF-ODE from max_t -> eps_t
    Then compare to initial structure.
    """
    pos0, R0, sequence = load_initial_structure(init_npz, sequence)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n = len(sequence)

    results_dir = Path(results_dir)
    with open(results_dir / "params.txt", "r") as f:
        params = f.read()
    params = eval(params)

    # Resolve model and SDEs
    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )
    score_model = load_model(ckpt_path, model_config_path)
    score_model.eval()
    sdes = load_sdes(model_config_path=model_config_path, cache_so3_dir=cache_so3_dir)

    context, batch = build_chemgraph(sequence, pos0, R0)

    # Forward PF-ODE (deterministic)
    guiding_pos, guiding_rot, _ = forward_probability_flow_trajectory(
        sdes=sdes,
        batch=batch,
        score_model=score_model,
        N=N,
        eps_t=eps_t,
        max_t=max_t,
        device=device,
        method=method,
    )

    # Reverse PF-ODE
    sampled_positions, sampled_orientations, t_b, excess_work = guided_reverse_integration(
        guiding_positions=guiding_pos,
        guiding_orientations=guiding_rot,
        context=context,
        sdes=sdes,
        score_model=score_model,
        N=N,
        eps_t=eps_t,
        max_t=max_t,
        method=method,
        device=device,
        sequence_length=n,
        batch_size=batch_size,
        params=params,
    )

    pos_rec = reshape_positions(sampled_positions[-1].detach().cpu(), batch_size, n)
    R_rec = reshape_orientations(sampled_orientations[-1].detach().cpu(), batch_size, n)


    data_dir = "/lustre/groups/bauer/code/thoeppe/bioemu/protein_data/transition_states"

    transition_classifier = TransitionClassifier(
        npz_path=data_dir + "/state_analysis_summary.npz",
        n_neighbors=1,
        use_rotations=True,
    )

    result = transition_classifier.classify(
        new_ca_positions=pos_rec,
        new_node_orientations=R_rec,
        q_transition_low=0.4,
        q_transition_high=0.6,
    )

    print("committor:", result["committor"])
    print("transition-like indices:", np.where(result["is_transition"])[0])
    print("macrostate labels:", result["macrostate_from_state"])
    print("percentage of transition-like states:", np.mean(result["is_transition"]))

    np.savez(
        f"{results_dir}/batch_{N}_{batch_size}.npz",
        pos=sampled_positions[-1].detach().cpu().numpy(),
        node_orientations=sampled_orientations[-1].detach().cpu().numpy(),
        macrostate_labels=result["macrostate_from_state"],
    )

    # Compute and report mean Wasserstein distance between bond length distributions
    # only compute for data with macrostate label 2
    macrostate_label_2 = np.where(result["macrostate_from_state"] == 2)[0]
    pos_rec_macrostate_label_2 = pos_rec[macrostate_label_2]
    mean_wd = compute_mean_wasserstein_bond_distance(
        edge_index=context.edge_index,
        positions_a=pos_rec_macrostate_label_2,
        positions_b=pos0,
        num_edges=None,
    )
    print(f"Mean Wasserstein distance over bonds (samples vs initial): {mean_wd:.6f}")
    try:
        with open(f"{results_dir}/wd_{N}_{batch_size}.txt", "w") as f:
            f.write(str(mean_wd))
    except Exception:
        pass


if __name__ == "__main__":
    import logging as _logging
    import fire as _fire

    _logging.basicConfig(level=_logging.INFO)
    _fire.Fire(main)


