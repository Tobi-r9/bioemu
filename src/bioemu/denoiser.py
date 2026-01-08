# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import cast, Callable

import numpy as np
import torch
from torch_geometric.data.batch import Batch

from .chemgraph import ChemGraph
from .sde_lib import SDE, CosineVPSDE
from .so3_sde import SO3SDE, apply_rotvec_to_rotmat
from .mew_utils import gaussian_kde_score_batched, align_points, so3_gaussian_kde_score_batched
TwoBatches = tuple[Batch, Batch]


class EulerMaruyamaPredictor:
    """Euler-Maruyama predictor."""

    def __init__(
        self,
        *,
        corruption: SDE,
        noise_weight: float = 1.0,
        marginal_concentration_factor: float = 1.0,
    ):
        """
        Args:
            noise_weight: A scalar factor applied to the noise during each update. The parameter controls the stochasticity of the integrator. A value of 1.0 is the
            standard Euler Maruyama integration scheme whilst a value of 0.0 is the probability flow ODE.
            marginal_concentration_factor: A scalar factor that controls the concentration of the sampled data distribution. The sampler targets p(x)^{MCF} where p(x)
            is the data distribution. A value of 1.0 is the standard Euler Maruyama / probability flow ODE integration.

            See feynman/projects/diffusion/sampling/samplers_readme.md for more details.

        """
        self.corruption = corruption
        self.noise_weight = noise_weight
        self.marginal_concentration_factor = marginal_concentration_factor

    def reverse_drift_and_diffusion(
        self, *, x: torch.Tensor, t: torch.Tensor, batch_idx: torch.LongTensor, score: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        score_weight = 0.5 * self.marginal_concentration_factor * (1 + self.noise_weight**2)
        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        drift = drift - diffusion**2 * score * score_weight
        return drift, diffusion

    def update_given_drift_and_diffusion(
        self,
        *,
        x: torch.Tensor,
        dt: torch.Tensor,
        drift: torch.Tensor,
        diffusion: torch.Tensor,
    ) -> TwoBatches:
        z = torch.randn_like(drift)

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        if isinstance(self.corruption, SO3SDE):
            mean = apply_rotvec_to_rotmat(x, drift * dt, tol=self.corruption.tol)
            sample = apply_rotvec_to_rotmat(
                mean,
                self.noise_weight * diffusion * torch.sqrt(dt.abs()) * z,
                tol=self.corruption.tol,
            )
        else:
            mean = x + drift * dt
            sample = mean + self.noise_weight * diffusion * torch.sqrt(dt.abs()) * z
        return sample, mean

    def update_given_score(
        self,
        *,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
        score: torch.Tensor,
    ) -> TwoBatches:

        # Set up different coefficients and terms.
        drift, diffusion = self.reverse_drift_and_diffusion(
            x=x, t=t, batch_idx=batch_idx, score=score
        )

        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(
            x=x,
            dt=dt,
            drift=drift,
            diffusion=diffusion,
        )

    def forward_sde_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        batch_idx: torch.LongTensor,
    ) -> TwoBatches:
        """Update to next step using either special update for SDEs on SO(3) or standard update.
        Handles both SO(3) and Euclidean updates."""

        drift, diffusion = self.corruption.sde(x=x, t=t, batch_idx=batch_idx)
        # Update to next step using either special update for SDEs on SO(3) or standard update.
        return self.update_given_drift_and_diffusion(x=x, dt=dt, drift=drift, diffusion=diffusion)


def get_score(
    batch: ChemGraph, sdes: dict[str, SDE], score_model: torch.nn.Module, t: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Calculate predicted score for the batch.

    Args:
        batch: Batch of corrupted data.
        sdes: SDEs.
        score_model: Score model.  The score model is parametrized to predict a multiple of the score.
          This function converts the score model output to a score.
        t: Diffusion timestep. Shape [batch_size,]
    """
    # Some models expect batched t shaped [num_graphs, 1]
    tt = t.view(-1)
    if tt.shape[0] == 1 and hasattr(batch, "num_graphs"):
        tt = tt.expand(batch.num_graphs)
    tmp = score_model(batch, tt)
    # Score is in axis angle representation [N,3] (vector is along axis of rotation, vector length
    # is rotation angle in radians).
    assert isinstance(sdes["node_orientations"], SO3SDE)
    node_orientations_score = (
        tmp["node_orientations"]
        * sdes["node_orientations"].get_score_scaling(t, batch_idx=batch.batch)[:, None]
    )

    # Score model is trained to predict score * std, so divide by std to get the score.
    _, pos_std = sdes["pos"].marginal_prob(
        x=torch.ones_like(tmp["pos"]),
        t=t,
        batch_idx=batch.batch,
    )
    pos_score = tmp["pos"] / pos_std

    return {"node_orientations": node_orientations_score, "pos": pos_score}


def heun_denoiser(
    *,
    sdes: dict[str, SDE],
    N: int,
    eps_t: float,
    max_t: float,
    device: torch.device,
    batch: Batch,
    score_model: torch.nn.Module,
    noise: float,
) -> ChemGraph:
    """Sample from prior and then denoise."""

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    assert isinstance(sdes["node_orientations"], torch.nn.Module)  # shut up mypy
    sdes["node_orientations"] = sdes["node_orientations"].to(device)
    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )

    ts_min = 0.0
    ts_max = 1.0
    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    fields = list(sdes.keys())
    predictors = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    batch_size = batch.num_graphs

    for i in range(N):
        # Set the timestep
        t = torch.full((batch_size,), timesteps[i], device=device)
        t_next = t + dt  # dt is negative; t_next is slightly less noisy than t.

        # Select temporarily increased noise level t_hat.
        # To be more general than Algorithm 2 in Karras et al. we select a time step between the
        # current and the previous t.
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)

        score = get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        # First-order denoising step from t_hat to t_next.
        drift_hat = {}
        for field in fields:
            drift_hat[field], _ = predictors[field].reverse_drift_and_diffusion(
                x=batch_hat[field], t=t_hat, batch_idx=batch.batch, score=score[field]
            )

        for field in fields:
            batch[field] = predictors[field].update_given_drift_and_diffusion(
                x=batch_hat[field],
                dt=(t_next - t_hat)[0],
                drift=drift_hat[field],
                diffusion=0.0,
            )[0]

        # Apply 2nd order correction.
        if t_next[0] > 0.0:
            score = get_score(batch=batch, t=t_next, score_model=score_model, sdes=sdes)

            drifts = {}
            avg_drift = {}
            for field in fields:
                drifts[field], _ = predictors[field].reverse_drift_and_diffusion(
                    x=batch[field], t=t_next, batch_idx=batch.batch, score=score[field]
                )

                avg_drift[field] = (drifts[field] + drift_hat[field]) / 2
            for field in fields:
                batch[field] = (
                    0.0
                    + predictors[field].update_given_drift_and_diffusion(
                        x=batch_hat[field],
                        dt=(t_next - t_hat)[0],
                        drift=avg_drift[field],
                        diffusion=0.0,
                    )[0]
                )

    return batch


def _t_from_lambda(sde: CosineVPSDE, lambda_t: torch.Tensor) -> torch.Tensor:
    """
    Used for DPMsolver. https://arxiv.org/abs/2206.00927 Appendix Section D.4
    """
    f_lambda = -1 / 2 * torch.log(torch.exp(-2 * lambda_t) + 1)
    exponent = f_lambda + torch.log(torch.cos(torch.tensor(np.pi * sde.s / 2 / (1 + sde.s))))
    t_lambda = 2 * (1 + sde.s) / np.pi * torch.acos(torch.exp(exponent)) - sde.s
    return t_lambda


def dpm_solver(
    sdes: dict[str, SDE],
    batch: Batch,
    N: int,
    score_model: torch.nn.Module,
    max_t: float,
    eps_t: float,
    device: torch.device,
    record_grad_steps: set[int] = set(),
    noise: float = 0.0,
) -> ChemGraph:

    """
    Implements the DPM solver for the VPSDE, with the Cosine noise schedule.
    Following this paper: https://arxiv.org/abs/2206.00927 Algorithm 1 DPM-Solver-2.
    DPM solver is used only for positions, not node orientations.
    """
    grad_is_enabled = torch.is_grad_enabled()
    assert isinstance(batch, ChemGraph)
    assert max_t < 1.0

    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        # permits unit-testing with dummy model
        score_model = score_model.to(device)
    pos_sde = sdes["pos"]
    assert isinstance(pos_sde, CosineVPSDE)

    batch = batch.replace(
        pos=sdes["pos"].prior_sampling(batch.pos.shape, device=device),
        node_orientations=sdes["node_orientations"].prior_sampling(
            batch.node_orientations.shape, device=device
        ),
    )
    batch = cast(ChemGraph, batch)  # help out mypy/linter

    so3_sde = sdes["node_orientations"]
    assert isinstance(so3_sde, SO3SDE)
    so3_sde.to(device)

    timesteps = torch.linspace(max_t, eps_t, N, device=device)
    dt = -torch.tensor((max_t - eps_t) / (N - 1)).to(device)
    ts_min = 0.0
    ts_max = 1.0
    fields = list(sdes.keys())
    noisers = {
        name: EulerMaruyamaPredictor(
            corruption=sde, noise_weight=1.0, marginal_concentration_factor=1.0
        )
        for name, sde in sdes.items()
    }
    for i in range(N - 1):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)
        t_hat = t - noise * dt if (i > 0 and t[0] > ts_min and t[0] < ts_max) else t

        # Apply noise.
        vals_hat = {}
        for field in fields:
            vals_hat[field] = noisers[field].forward_sde_step(
                x=batch[field], t=t, dt=(t_hat - t)[0], batch_idx=batch.batch
            )[0]
        batch_hat = batch.replace(**vals_hat)

        # Evaluate score
        with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
            score = get_score(batch=batch_hat, t=t_hat, score_model=score_model, sdes=sdes)

        # t_{i-1} in the algorithm is the current t
        batch_idx = batch_hat.batch
        alpha_t, sigma_t = pos_sde.mean_coeff_and_std(x=batch.pos, t=t_hat, batch_idx=batch_idx)
        lambda_t = torch.log(alpha_t / sigma_t)
        alpha_t_next, sigma_t_next = pos_sde.mean_coeff_and_std(
            x=batch.pos, t=t + dt, batch_idx=batch_idx
        )
        lambda_t_next = torch.log(alpha_t_next / sigma_t_next)

        # t+dt < t_hat, lambad_t_next > lambda_t
        h_t = lambda_t_next - lambda_t

        # For a given noise schedule (cosine is what we use), compute the intermediate t_lambda
        lambda_t_middle = (lambda_t + lambda_t_next) / 2
        t_lambda = _t_from_lambda(sde=pos_sde, lambda_t=lambda_t_middle)

        # t_lambda has all the same components
        t_lambda = torch.full((batch.num_graphs,), t_lambda[0][0], device=device)

        alpha_t_lambda, sigma_t_lambda = pos_sde.mean_coeff_and_std(
            x=batch.pos, t=t_lambda, batch_idx=batch_idx
        )
        # Note in the paper the algorithm uses noise instead of score, but we use score.
        # So the formulation is slightly different in the prefactor.
        u = (
            alpha_t_lambda / alpha_t * batch_hat.pos
            + sigma_t_lambda * sigma_t * (torch.exp(h_t / 2) - 1) * score["pos"]
        )

        # Update positions to the intermediate timestep t_lambda
        batch_u = batch.replace(pos=u)

        # Get node orientation at t_lambda

        # Denoise from t to t_lambda
        assert score["node_orientations"].shape == (u.shape[0], 3)
        assert batch.node_orientations.shape == (u.shape[0], 3, 3)
        so3_predictor = EulerMaruyamaPredictor(
            corruption=so3_sde, noise_weight=0.0, marginal_concentration_factor=1.0
        )
        drift, _ = so3_predictor.reverse_drift_and_diffusion(
            x=batch_hat.node_orientations,
            score=score["node_orientations"],
            t=t_hat,
            batch_idx=batch_idx,
        )
        sample, _ = so3_predictor.update_given_drift_and_diffusion(
            x=batch_hat.node_orientations,
            drift=drift,
            diffusion=0.0,
            dt=t_lambda[0] - t_hat[0],
        )  # dt is negative, diffusion is 0
        assert sample.shape == (u.shape[0], 3, 3)
        batch_u = batch_u.replace(node_orientations=sample)

        # Correction step
        # Evaluate score at updated pos and node orientations
        with torch.set_grad_enabled(grad_is_enabled and (i in record_grad_steps)):
            score_u = get_score(batch=batch_u, t=t_lambda, sdes=sdes, score_model=score_model)

        pos_next = (
            alpha_t_next / alpha_t * batch_hat.pos
            + sigma_t_next * sigma_t_lambda * (torch.exp(h_t) - 1) * score_u["pos"]
        )
        batch_next = batch.replace(pos=pos_next)

        assert score_u["node_orientations"].shape == (u.shape[0], 3)

        # Try a 2nd order correction
        dt_hat = t + dt - t_hat
        node_score = (
            score_u["node_orientations"]
            + 0.5
            * (score_u["node_orientations"] - score["node_orientations"])
            / (t_lambda[0] - t_hat[0])
            * dt_hat[0]
        )
        drift, diffusion = so3_predictor.reverse_drift_and_diffusion(
            x=batch_u.node_orientations,
            score=node_score,
            t=t_lambda,
            batch_idx=batch_idx,
        )
        sample, _ = so3_predictor.update_given_drift_and_diffusion(
            x=batch_hat.node_orientations,
            drift=drift,
            diffusion=0.0,
            dt=dt_hat[0],
        )  # dt is negative, diffusion is 0
        batch = batch_next.replace(node_orientations=sample)

    return batch


@torch.no_grad()
def reverse_probability_flow_trajectory(
    sdes: dict[str, SDE],
    batch: ChemGraph,
    score_model: torch.nn.Module,
    N: int,
    eps_t: float,
    max_t: float,
    device: torch.device,
    method: str = "euler",
    noise_weight: float = 0.0,
    guiding_samples: dict[str, dict[float, torch.Tensor]] = {"pos": {}, "node_orientations": {}},
    guiding_strength_func: Callable[[float], float] = lambda t: t,
    bandwidth_func: Callable[[float], float] = lambda t: 1,
    guiding_strength_rot_func: Callable[[float], float] = lambda t: t,
    bandwidth_rot_func: Callable[[float], float] = lambda t: 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Reverse-time integration (probability flow ODE; noise_weight=0) from t=max_t to t=eps_t.
    Returns trajectory of pos and node_orientations.
    """
    assert "pos" in sdes and "node_orientations" in sdes
    pos_sde = sdes["pos"]
    so3_sde = sdes["node_orientations"]

    # Ensure pyg Batch
    if not isinstance(batch, Batch):
        batch = Batch.from_data_list([batch])
    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        score_model = score_model.to(device)
    assert isinstance(so3_sde, torch.nn.Module)
    so3_sde = so3_sde.to(device)

    # Avoid singularities at t=1 for CosineVPSDE (beta -> inf)
    max_t_eff = min(float(max_t), 1.0 - 1e-5)
    timesteps = torch.linspace(max_t_eff, float(eps_t), N, device=device)
    dt = -torch.tensor((max_t_eff - float(eps_t)) / (N - 1), device=device)

    pos_traj = [batch.pos.clone().detach().cpu()]
    node_traj = [batch.node_orientations.clone().detach().cpu()]
    d = len(batch.sequence[0])
    excess_work = {
        "pos": torch.tensor(0.0, device=device),
        "node_orientations": torch.tensor(0.0, device=device),
    }
    print(d)

    predictors = {
        name: EulerMaruyamaPredictor(corruption=sde, noise_weight=noise_weight, marginal_concentration_factor=1.0)
        for name, sde in sdes.items()
    }
    for i in range(N - 1):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)
        score = get_score(batch=batch, sdes=sdes, score_model=score_model, t=t)

        # Compute reverse drift terms
        for field in ["pos", "node_orientations"]:
            if guiding_samples[field]:
                closest_key = min(guiding_samples[field].keys(), key=lambda x: abs(x - timesteps[i]))
                guiding_sample = guiding_samples[field][closest_key].to(device)
                if field == "pos":
                    K, M = batch[field].shape[0] // d, guiding_sample.shape[0] // d
                    guiding_sample = align_points(batch[field], guiding_sample, sequence_length=d)
                    kde_score = gaussian_kde_score_batched(
                        batch[field].view(K, -1),
                        guiding_sample.view(K, M, -1),
                        bandwidth_func(timesteps[i].item()),
                    ).view(-1, 3)
                    guiding_score = guiding_strength_func(timesteps[i].item()) * kde_score
                    score[field] = score[field] + guiding_score
                elif field == "node_orientations":
                    kde_score_rot = so3_gaussian_kde_score_batched(
                        R=batch[field],
                        guiding_samples=guiding_sample,
                        sequence_length=d,
                        bandwidth=bandwidth_rot_func(timesteps[i].item()),
                    )
                    guiding_score_rot = guiding_strength_rot_func(timesteps[i].item()) * kde_score_rot
                    score[field] = score[field] + guiding_score_rot
            drift, diffusion = predictors[field].reverse_drift_and_diffusion(
                x=batch[field], t=t, batch_idx=batch.batch, score=score[field]
            )
            if guiding_samples[field] and field == "pos":
                w = diffusion / 2
                work_norm = (w * guiding_score.view(-1, 3)**2).sum(dim=-1)
                excess_work[field] = excess_work[field] + work_norm.mean()
            if guiding_samples[field] and field == "node_orientations":
                w = diffusion / 2
                work_norm = (w * guiding_score_rot**2).sum(dim=-1)
                excess_work[field] = excess_work[field] + work_norm.mean()
            if method == "euler":
                batch[field] = predictors[field].update_given_drift_and_diffusion(
                    x=batch[field], dt=dt, drift=drift, diffusion=diffusion
                )[0]
            elif method == "heun":
                # Heun (deterministic): average drifts
                x_pred = predictors[field].update_given_drift_and_diffusion(
                    x=batch[field], dt=dt, drift=drift, diffusion=torch.tensor(0.0, device=device)
                )[0]
                drift2, _ = predictors[field].reverse_drift_and_diffusion(
                    x=x_pred, t=t + dt, batch_idx=batch.batch, score=score[field]
                )
                avg_drift = 0.5 * (drift + drift2)
                batch[field] = predictors[field].update_given_drift_and_diffusion(
                    x=batch[field], dt=dt, drift=avg_drift, diffusion=torch.tensor(0.0, device=device)
                )[0]
            else:
                raise ValueError("method must be 'euler' or 'heun'")

        pos_traj.append(batch.pos.clone().detach().cpu())
        node_traj.append(batch.node_orientations.clone().detach().cpu())

    return torch.stack(pos_traj), torch.stack(node_traj), timesteps, excess_work


@torch.no_grad()
def forward_probability_flow_trajectory(
    sdes: dict[str, SDE],
    batch: ChemGraph,
    score_model: torch.nn.Module,
    N: int,
    eps_t: float,
    max_t: float,
    device: torch.device,
    method: str = "euler",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Deterministic forward-time probability flow ODE from t=eps_t to t=max_t.
    Returns trajectory of pos and node_orientations.
    """
    assert "pos" in sdes and "node_orientations" in sdes
    pos_sde = sdes["pos"]
    so3_sde = sdes["node_orientations"]

    if not isinstance(batch, Batch):
        batch = Batch.from_data_list([batch])
    batch = batch.to(device)
    if isinstance(score_model, torch.nn.Module):
        score_model = score_model.to(device)
    assert isinstance(so3_sde, torch.nn.Module)
    so3_sde = so3_sde.to(device)

    max_t_eff = min(float(max_t), 1.0 - 1e-5)
    eps_t_eff = max(float(eps_t), 1e-5)
    timesteps = torch.linspace(eps_t_eff, max_t_eff, N, device=device)
    dt = torch.tensor((max_t_eff - eps_t_eff) / (N - 1), device=device)

    pos_traj = {timesteps[0].item(): batch.pos.clone()}
    node_traj = {timesteps[0].item(): batch.node_orientations.clone()}

    predictors = {
        name: EulerMaruyamaPredictor(corruption=sde, noise_weight=0.0, marginal_concentration_factor=1.0)
        for name, sde in sdes.items()
    }

    for i in range(N - 1):
        t = torch.full((batch.num_graphs,), timesteps[i], device=device)
        score = get_score(batch=batch, sdes=sdes, score_model=score_model, t=t)

        for field in ["pos", "node_orientations"]:
            drift, _ = predictors[field].reverse_drift_and_diffusion(
                x=batch[field], t=t, batch_idx=batch.batch, score=score[field]
            )
            if method == "euler":
                batch[field] = predictors[field].update_given_drift_and_diffusion(
                    x=batch[field], dt=dt, drift=drift, diffusion=torch.tensor(0.0, device=device)
                )[0]
            elif method == "heun":
                x_pred = predictors[field].update_given_drift_and_diffusion(
                    x=batch[field], dt=dt, drift=drift, diffusion=torch.tensor(0.0, device=device)
                )[0]
                drift2, _ = predictors[field].reverse_drift_and_diffusion(
                    x=x_pred, t=t + dt, batch_idx=batch.batch, score=score[field]
                )
                avg_drift = 0.5 * (drift + drift2)
                batch[field] = predictors[field].update_given_drift_and_diffusion(
                    x=batch[field], dt=dt, drift=avg_drift, diffusion=torch.tensor(0.0, device=device)
                )[0]
            else:
                raise ValueError("method must be 'euler' or 'heun'")

        pos_traj[timesteps[i+1].item()] = batch.pos.clone()
        node_traj[timesteps[i+1].item()] = batch.node_orientations.clone()

    return pos_traj, node_traj, timesteps
