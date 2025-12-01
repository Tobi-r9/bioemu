"""
CLI to integrate a provided structure forward under the SDE toward the prior,
returning and saving the full trajectory.

Input must provide coarse-grained frames: positions in nm and node orientations
as rotation matrices with shapes (num_residues, 3) and (num_residues, 3, 3).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .convert_chemgraph import save_pdb_and_xtc
from .denoiser import forward_integrate_to_prior
from .model_utils import load_sdes, maybe_download_checkpoint
from .seq_io import parse_sequence


logger = logging.getLogger(__name__)


@torch.no_grad()
def main(
    sequence: str | Path,
    init_npz: str | Path,
    output_dir: str | Path,
    N: int = 200,
    eps_t: float = 1e-3,
    max_t: float = 1.0,
    method: str = "euler",
    deterministic: bool = False,
    model_name: str | None = "bioemu-v1.1",
    model_config_path: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    filter_samples: bool = False,
) -> None:
    """
    Integrate an input structure toward the prior and save the trajectory.

    Args:
        sequence: Amino acid sequence or path to FASTA/A3M containing the sequence.
        init_npz: Path to .npz with keys 'pos' (nm) and 'node_orientations' (rotmats).
        output_dir: Output directory for PDB/XTC and numpy dumps.
        N: Number of time steps (including start and end).
        eps_t: Start time (>0).
        max_t: End time (≤1.0).
        method: 'euler' or 'heun'.
        model_name: Optional pretrained model name to resolve model config (only the corruption config is used).
        model_config_path: Alternatively provide explicit model config path for SDE definitions.
        cache_so3_dir: Optional directory for SO(3) precomputations used by the SDE.
        filter_samples: Whether to filter unphysical frames before writing XTC.
    """

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model config for SDE definitions.
    _, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=None, model_config_path=model_config_path
    )
    sdes = load_sdes(model_config_path=model_config_path, cache_so3_dir=cache_so3_dir)

    # Parse sequence (extract from file if necessary)
    sequence = parse_sequence(sequence)
    n = len(sequence)

    # Load initial coarse-grained frames
    init_npz = Path(init_npz)
    with np.load(init_npz, allow_pickle=False) as data:
        pos = torch.tensor(data["pos"])  # shape (..., N, 3) or (N, 3)
        node_orientations = torch.tensor(data["node_orientations"])  # (..., N, 3, 3)

    if pos.dim() == 3:
        # If batched, take the first frame
        pos = pos[0]
    if node_orientations.dim() == 4:
        node_orientations = node_orientations[0]

    assert pos.shape == (n, 3), f"pos has shape {tuple(pos.shape)} but expected ({n}, 3)"
    assert node_orientations.shape == (
        n,
        3,
        3,
    ), f"node_orientations has shape {tuple(node_orientations.shape)} but expected ({n}, 3, 3)"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(
        f"Integrating forward to prior with {N=} {eps_t=} {max_t=} {method=} on {device}"
    )

    pos_traj, node_orientations_traj, t_values = forward_integrate_to_prior(
        sdes=sdes,
        pos=pos,
        node_orientations=node_orientations,
        N=N,
        eps_t=eps_t,
        max_t=max_t,
        device=device,
        method=method,
        deterministic=deterministic,
    )

    # Save raw arrays
    np.savez(
        output_dir / "forward_traj.npz",
        pos=pos_traj.cpu().numpy(),
        node_orientations=node_orientations_traj.cpu().numpy(),
        t=t_values.cpu().numpy(),
        sequence=sequence,
    )

    # Save as PDB/XTC
    logger.info("Writing PDB and XTC...")
    save_pdb_and_xtc(
        pos_nm=pos_traj.unsqueeze(0),
        node_orientations=node_orientations_traj.unsqueeze(0),
        sequence=sequence,
        topology_path=output_dir / "topology.pdb",
        xtc_path=output_dir / "forward_traj.xtc",
        filter_samples=filter_samples,
    )
    logger.info(f"Done. Outputs written to {output_dir}")


if __name__ == "__main__":
    import logging as _logging
    import fire as _fire

    _logging.basicConfig(level=_logging.INFO)
    _fire.Fire(main)


