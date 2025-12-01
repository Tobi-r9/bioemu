from pathlib import Path

import numpy as np
import torch

from .denoiser import forward_probability_flow_trajectory
from .model_utils import load_model, load_sdes, maybe_download_checkpoint
from .sample_utils import (
    build_chemgraph,
    reshape_positions,
    reshape_orientations,
    plot_bond_distance_histograms,
    guided_reverse_integration,
    load_initial_structure,
)
from .transition_states import TransitionClassifier
from pathlib import Path

import functools
from skopt import gp_minimize
from skopt.space import Real

import os

# torch.manual_seed(0); np.random.seed(0); random.seed(0)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def run_gp_optimization(gamma, guiding_pos, guiding_rot, context, sdes, score_model, N, eps_t, max_t, method, device, sequence_length, batch_size, use_rotations, n_calls):

    print("start optimization")
    gp_optimization_function = functools.partial(
        optimization_function,
        guiding_pos=guiding_pos,
        guiding_rot=guiding_rot,
        context=context,
        sdes=sdes,
        score_model=score_model,
        N=N,
        eps_t=eps_t,
        max_t=max_t,
        method=method,
        device=device,
        sequence_length=sequence_length,
        batch_size=batch_size,
        gamma=gamma,
        use_rotations=use_rotations,
    )
    result = gp_minimize(
        gp_optimization_function,
        [
            Real(1, 5, name="alpha_s"),
            Real(19, 20, name="kappa_s"),
            Real(0.25, 0.75, name="beta_s"),
            Real(0.1, 1.5, name="alpha_b"),
            Real(19, 20, name="kappa_b"),
            Real(0.25, 0.75, name="beta_b"),
        ],
        # [
        #     Real(2.0, 3.0, name="alpha_s"),
        #     Real(0.2, 0.75, name="kappa_s"),
        #     Real(0.0, 0.25, name="beta_s"),
        #     Real(0.25, 1.0, name="alpha_b"),
        #     Real(0.2, 0.75, name="kappa_b"),
        #     Real(0.5, 1.5, name="beta_b"),
        # ],
        n_calls=n_calls,
        verbose=True,
    )
    return result


def optimization_function(
    params,
    guiding_pos,
    guiding_rot,
    context,
    sdes,
    score_model,
    N,
    eps_t,
    max_t,
    method,
    device,
    sequence_length,
    batch_size,
    gamma,
    use_rotations=False,
):

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
        sequence_length=sequence_length,
        batch_size=batch_size,
        params=params,
    )

    pos_rec = reshape_positions(sampled_positions[-1].detach().cpu(), batch_size, sequence_length)
    R_rec = reshape_orientations(sampled_orientations[-1].detach().cpu(), batch_size, sequence_length)

    data_dir = "/lustre/groups/bauer/code/thoeppe/bioemu/protein_data/transition_states"

    transition_classifier = TransitionClassifier(
        npz_path=data_dir + "/state_analysis_summary.npz",
        n_neighbors=1,
        use_rotations=use_rotations,
    )

    result = transition_classifier.classify(
        new_ca_positions=pos_rec,
        new_node_orientations=R_rec,
        q_transition_low=0.45,
        q_transition_high=0.55,
    )
    guiding_score = np.mean(result["is_transition"])
    excess_work = gamma * (1 / N) * excess_work["pos"].item()
    print(f"guiding_score: {guiding_score}, excess_work: {excess_work}")
    return (1 - guiding_score) + excess_work


@torch.no_grad()
def main(
    sequence: str | Path,
    init_npz: str | Path,
    batch_size: int = 64,
    N: int = 200,
    eps_t: float = 1e-3,
    max_t: float = 1.0,
    method: str = "euler",
    model_name: str | None = "bioemu-v1.1",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    use_rotations: bool = False,
    n_calls: int = 10,
    gamma: float = 1.0,
    save_dir: str | Path = "path_guidance_samples",
) -> None:
    """
    Deterministic probability-flow roundtrip check using the trained score model:
    - Forward PF-ODE from eps_t -> max_t
    - Reverse PF-ODE from max_t -> eps_t
    Then compare to initial structure.
    """
    pos0, R0, sequence = load_initial_structure(init_npz, sequence)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sequence_length = len(sequence)

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

    result = run_gp_optimization(
        gamma=gamma,
        guiding_pos=guiding_pos,
        guiding_rot=guiding_rot,
        context=context,
        sdes=sdes,
        score_model=score_model,
        N=N,
        eps_t=eps_t,
        max_t=max_t,
        method=method,
        device=device,
        sequence_length=sequence_length,
        batch_size=batch_size,
        use_rotations=use_rotations,
        n_calls=n_calls,
    )

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
        sequence_length=sequence_length,
        batch_size=batch_size,
        params=result.x,
    )
    save_dir = f"{save_dir}/{gamma}_{batch_size}_{N}_{eps_t}_{max_t}_{method}_{use_rotations}"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        f"{save_dir}/samples.npz",
        pos=sampled_positions[-1].detach().cpu().numpy(),
        node_orientations=sampled_orientations[-1].detach().cpu().numpy(),
    )

    # save result.x to txt file
    with open(f"{save_dir}/params.txt", "w") as f:
        f.write(str(result.x))



if __name__ == "__main__":
    import logging as _logging
    import fire as _fire

    _logging.basicConfig(level=_logging.INFO)
    _fire.Fire(main)
