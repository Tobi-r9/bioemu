import torch
from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Batch

from .chemgraph import ChemGraph
from .denoiser import reverse_probability_flow_trajectory
from .sample import get_context_chemgraph
from .seq_io import parse_sequence

from typing import Callable

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

def load_initial_structure(init_npz: str | Path, sequence: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = parse_sequence(sequence)
    n = len(sequence)
        # Load initial structure
    with np.load(init_npz, allow_pickle=False) as data:
        print(data.keys())
        pos0 = torch.tensor(data["positions"])  # (N,3) or (B,N,3)
        R0 = torch.tensor(data["node_orientations"])  # (N,3,3) or (B,N,3,3)
    if pos0.dim() == 2:
        pos0 = pos0.unsqueeze(0)
    if R0.dim() == 3:
        R0 = R0.unsqueeze(0)
    assert pos0.shape[1:] == (n, 3)
    assert R0.shape[1:] == (n, 3, 3)
    return pos0, R0, sequence

def build_chemgraph(sequence: str, positions: torch.Tensor, orientations: torch.Tensor) -> tuple[ChemGraph, Batch]:
    context = get_context_chemgraph(sequence=sequence)
    batch = build_batch_from_arrays(context, positions, orientations)
    return context, batch


def guided_reverse_integration(
    guiding_positions: dict[float, torch.Tensor],
    guiding_orientations: dict[float, torch.Tensor],
    context: ChemGraph,
    sdes: dict[str, torch.nn.Module],
    score_model: torch.nn.Module,
    params: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ],
    N: int = 200,
    eps_t: float = 1e-3,
    max_t: float = 1.0,
    method: str = "euler",
    device: torch.device = None,
    sequence_length: int = None,
    batch_size: int = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:

    if batch_size is None or sequence_length is None:
        raise ValueError("Need sequence_length and batch_size to sample from the prior.")
    pos_forward = sdes["pos"].prior_sampling((batch_size, sequence_length, 3), device=device)
    rot_forward = sdes["node_orientations"].prior_sampling((batch_size, sequence_length, 3, 3), device=device)

    if len(params) != 12:
        raise ValueError("params must contain 12 floats: 6 for pos, 6 for rotations.")
    (
        alpha_s,
        kappa_s,
        beta_s,
        alpha_b,
        kappa_b,
        beta_b,
        alpha_s_r,
        kappa_s_r,
        beta_s_r,
        alpha_b_r,
        kappa_b_r,
        beta_b_r,
    ) = (torch.tensor(p) for p in params)

    guiding_strength_func = lambda t: alpha_s * (1 - torch.sigmoid(kappa_s * (t - beta_s)))
    bandwidth_func = lambda t: alpha_b + (torch.sigmoid(kappa_b * (t - beta_b)))
    guiding_strength_rot_func = lambda t: alpha_s_r * (1 - torch.sigmoid(kappa_s_r * (t - beta_s_r)))
    bandwidth_rot_func = lambda t: alpha_b_r + (torch.sigmoid(kappa_b_r * (t - beta_b_r)))

    batch_f = build_batch_from_arrays(context, pos_forward, rot_forward)
    sampled_positions, sampled_orientations, t_b, excess_work = reverse_probability_flow_trajectory(
        sdes=sdes,
        batch=batch_f,
        score_model=score_model,
        N=N,
        eps_t=eps_t,
        max_t=max_t,
        device=device,
        method=method,
        guiding_samples={"pos": guiding_positions, "node_orientations": guiding_orientations},
        noise_weight=1.0,
        guiding_strength_func=guiding_strength_func,
        bandwidth_func=bandwidth_func,
        guiding_strength_rot_func=guiding_strength_rot_func,
        bandwidth_rot_func=bandwidth_rot_func,
    )

    return sampled_positions, sampled_orientations, t_b, excess_work