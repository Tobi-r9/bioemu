"""
CLI to check deterministic reversibility: integrate forward (probability flow style) and back,
and compare to the initial structure.
"""

from __future__ import annotations

import logging
from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch

from .chemgraph import ChemGraph
from .denoiser import (
    reverse_probability_flow_trajectory,
    forward_probability_flow_trajectory,
)
from .model_utils import load_model, load_sdes, maybe_download_checkpoint
from .sample import get_context_chemgraph
from .seq_io import parse_sequence

import torch, numpy as np, random, os
# torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  


logger = logging.getLogger(__name__)


def plot_bond_distance_histograms(
    edge_index: torch.Tensor,
    positions: torch.Tensor,
    num_edges: int = 10,
    bins: int = 30,
    title_prefix: str = "",
    file_name: str = "",
) -> None:
    """
    Plot histograms of bond distances for the provided positions and edges.

    Args:
        edge_index: Edge index tensor with shape [2, num_edges_total].
        positions: Tensor with shape [N, 3] or [T, N, 3]. When [N, 3] is passed,
            it is interpreted as a single frame.
        num_edges: Number of edges to visualize (truncated if graph has fewer edges).
        bins: Number of histogram bins.
        title_prefix: Text prepended to each subplot title.
    """

    if positions.dim() == 2:
        positions = positions.unsqueeze(0)
    elif positions.dim() > 3:
        leading = int(np.prod(positions.shape[:-2]))
        positions = positions.reshape(leading, positions.shape[-2], positions.shape[-1])
    positions = positions.detach().cpu()
    edge_index = edge_index.detach().cpu().long()

    total_edges = edge_index.shape[1]
    if total_edges == 0:
        logger.warning("No edges available to plot bond distances.")
        return

    num_edges = min(num_edges, total_edges)
    selected_edges = edge_index[:, :num_edges]

    num_cols = min(5, num_edges)
    num_rows = math.ceil(num_edges / num_cols)
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(4 * num_cols, 3 * num_rows),
        squeeze=False,
    )
    axes_flat = axes.flatten()

    for idx, (i, j) in enumerate(selected_edges.t()):
        dists = torch.linalg.norm(positions[:, i, :] - positions[:, j, :], dim=-1).numpy()
        ax = axes_flat[idx]
        ax.hist(dists, bins=bins, color="tab:blue", alpha=0.8)
        ax.set_title(f"{title_prefix} edge ({i.item()}, {j.item()})")
        ax.set_xlabel("distance (nm)")
        ax.set_ylabel("count")

    for ax in axes_flat[num_edges:]:
        ax.axis("off")

    fig.suptitle(title_prefix or "Bond distance histograms")
    fig.tight_layout()
    if file_name:
        plt.savefig(file_name)
    plt.close()


def build_batch_from_arrays(
    context: ChemGraph, pos: torch.Tensor, node_orientations: torch.Tensor
) -> Batch:
    """Construct a Batch of ChemGraphs that share the context embeddings."""
    data_list = [
        context.replace(pos=pos[i], node_orientations=node_orientations[i]) for i in range(pos.shape[0])
    ]
    return Batch.from_data_list(data_list)


def reshape_positions(frame: torch.Tensor, batch_size: int, n: int) -> torch.Tensor:
    return frame.reshape(batch_size, n, 3)


def reshape_orientations(frame: torch.Tensor, batch_size: int, n: int) -> torch.Tensor:
    return frame.reshape(batch_size, n, 3, 3)


@torch.no_grad()
def main(
    sequence: str | Path,
    init_npz: str | Path,
    N: int = 200,
    eps_t: float = 1e-3,
    max_t: float = 1.0,
    method: str = "heun",
    model_name: str | None = "bioemu-v1.1",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    atol_pos: float = 5e-4,
    atol_rot: float = 5e-4,
) -> None:
    """
    Deterministic probability-flow roundtrip check using the trained score model:
    - Forward PF-ODE from eps_t -> max_t
    - Reverse PF-ODE from max_t -> eps_t
    Then compare to initial structure.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sequence = parse_sequence(sequence)
    n = len(sequence)

    # Resolve model and SDEs
    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )
    print(ckpt_path, model_config_path)
    score_model = load_model(ckpt_path, model_config_path)
    score_model.eval()
    sdes = load_sdes(model_config_path=model_config_path, cache_so3_dir=cache_so3_dir)
    # Load initial structure
    with np.load(init_npz, allow_pickle=False) as data:
        pos0 = torch.tensor(data["pos"])  # (N,3) or (B,N,3)
        R0 = torch.tensor(data["node_orientations"])  # (N,3,3) or (B,N,3,3)
    if pos0.dim() == 2:
        pos0 = pos0.unsqueeze(0)
    if R0.dim() == 3:
        R0 = R0.unsqueeze(0)
    batch_size = pos0.shape[0]
    assert pos0.shape[1:] == (n, 3)
    assert R0.shape[1:] == (n, 3, 3)

    # Build ChemGraph context with embeddings
    context = get_context_chemgraph(sequence=sequence)
    batch0 = build_batch_from_arrays(context, pos0, R0)
    print(batch0)

    plot_bond_distance_histograms(
        edge_index=context.edge_index,
        positions=pos0,
        num_edges=10,
        title_prefix="Initial batch0",
        file_name="initial_batch0.png",
    )

    # Forward PF-ODE (deterministic)
    pos_f, R_f, t_f = forward_probability_flow_trajectory(
        sdes=sdes,
        batch=batch0,
        score_model=score_model,
        N=N,
        eps_t=eps_t,
        max_t=max_t,
        device=device,
        method=method,
    )

    # Sanity: ensure rotation changed across forward path (unless very small dt)
    # rot_change = torch.norm(R_f[-1].cpu() - R_f[0].cpu()).item()
    # pos_change = torch.norm(pos_f[-1].cpu() - pos_f[0].cpu()).item()
    # logger.info(f"forward changes: pos_change={pos_change:.3e} rot_change={rot_change:.3e}")

    # Reverse PF-ODE
    pos_max_key = max(pos_f.keys())
    R_max_key = max(R_f.keys())
    pos_forward = reshape_positions(pos_f[pos_max_key].detach().cpu(), batch_size, n)
    rot_forward = reshape_orientations(R_f[R_max_key].detach().cpu(), batch_size, n)
    batch_f = build_batch_from_arrays(context, pos_forward, rot_forward)
    pos_b, R_b, t_b = reverse_probability_flow_trajectory(
        sdes=sdes,
        batch=batch_f,
        score_model=score_model,
        N=N,
        eps_t=eps_t,
        max_t=max_t,
        device=device,
        method=method,
    )

    pos_rec = reshape_positions(pos_b[-1].detach().cpu(), batch_size, n)
    R_rec = reshape_orientations(R_b[-1].detach().cpu(), batch_size, n)

    plot_bond_distance_histograms(
        edge_index=context.edge_index,
        positions=pos_rec,
        num_edges=10,
        title_prefix="Reverse PF",
        file_name="reverse_pf_final.png",
    )

    # pos_err = torch.max(torch.abs(pos_rec - pos0.cpu())).item()
    # rot_err = torch.max(torch.abs(R_rec - R0.cpu())).item()
    pos_err = torch.norm(pos_rec - pos0.cpu(), dim=2).mean().item()
    rot_err = torch.norm(R_rec - R0.cpu(), dim=(2, 3)).mean().item()

    logger.info(f"reconstruction: pos_err={pos_err:.3e} rot_err={rot_err:.3e}")

    ok = (pos_err <= atol_pos) and (rot_err <= atol_rot)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    import logging as _logging
    import fire as _fire

    _logging.basicConfig(level=_logging.INFO)
    _fire.Fire(main)


