import torch
from .so3_sde import rot_mult, rot_transpose, rotmat_to_rotvec

@torch.no_grad()
def kabsch_torch_batched(P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """
    Computes the optimal rotation and translation to align two sets of points (P -> Q),
    and their RMSD, in a batched manner.
    :param P: A BxNx3 matrix of points
    :param Q: A BxNx3 matrix of points
    :return: A tuple containing the optimal rotation matrix, the optimal
             translation vector, and the RMSD.
    """
    assert (
        P.shape == Q.shape
    ), f"P and Q must have the same dimension, but found P: {P.shape} and Q: {Q.shape}"

    centroid_P = torch.mean(P, dim=1, keepdims=True)
    centroid_Q = torch.mean(Q, dim=1, keepdims=True)

    t = centroid_Q - centroid_P
    t = t.squeeze(1)
    p = P - centroid_P
    q = Q - centroid_Q

    H = torch.matmul(p.transpose(1, 2), q)
    U, S, Vt = torch.linalg.svd(H)
    d = torch.det(torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2)))
    flip = d < 0.0
    if flip.any().item():
        Vt[flip, -1] *= -1.0

    R = torch.matmul(Vt.transpose(1, 2), U.transpose(1, 2))
    # rmsd = torch.sqrt(torch.sum(torch.square(torch.matmul(p, R.transpose(1, 2)) - q), dim=(1, 2)) / P.shape[1])
    return torch.matmul(p, R.transpose(1, 2))


@torch.no_grad()
def align_points(x: torch.Tensor, guiding_points: torch.Tensor, sequence_length: int = 10) -> torch.Tensor:
    """
    Align points for guiding using Kabsch algorithm
    Args:
        x: tensor of shape (N, d)
        guiding_points: tensor of shape (N, M, d)
    Returns:
        aligned_points: tensor of shape (N, M, d)
    """
    guiding_points = guiding_points.view(-1, sequence_length, 3)
    x = x.view(-1, sequence_length, 3)
    N, M = x.size(0), guiding_points.size(0)
    guiding_points_repeated = guiding_points.repeat_interleave(N, dim=0)
    x_transposed = x.permute(1, 0, 2)
    x_repeated = x_transposed.repeat(1, M, 1)
    x_reorganized = x_repeated.permute(1, 0, 2).reshape(N * M, sequence_length, 3)
    guiding_samples_aligned = kabsch_torch_batched(guiding_points_repeated, x_reorganized)
    return guiding_samples_aligned


@torch.no_grad()
def gaussian_kde_score_batched(x: torch.Tensor, guiding_samples: torch.Tensor, bandwidth: float = 1) -> torch.Tensor:
    """
    Compute KDE score with batched guiding samples
    Args:
        x: tensor of shape (N, d)
        guiding_samples: tensor of shape (N, M, d)
        h: bandwidth parameter
    Returns:
        score: tensor of shape (N, d)
    """
    eps = 1e-20
    x_expanded = x.unsqueeze(1)
    diff = x_expanded - guiding_samples

    density = torch.exp(-(diff**2).sum(dim=2) / (2 * (bandwidth**2)))

    mask = torch.norm(density, dim=1) == 0
    if mask.any():
        density[mask] = torch.ones_like(density[mask]) * eps

    score = -(1 / (bandwidth**2)) * diff
    weighted_score = (score * density.unsqueeze(2)).sum(dim=1)
    denominator = density.sum(dim=1, keepdim=True)

    return weighted_score / (denominator + eps) 


@torch.no_grad()
def so3_gaussian_kde_score_batched(
    R: torch.Tensor,
    guiding_samples: torch.Tensor,
    sequence_length: int,
    bandwidth: float = 1.0,
    eps: float = 1e-20,
) -> torch.Tensor:
    """
    KDE score on SO(3) using right-trivialized log map (R @ Exp(v)).
    Args:
        R: rotations of shape (N, 3, 3) where N = batch_size * sequence_length.
        guiding_samples: guiding rotations of shape (M * sequence_length, 3, 3).
        sequence_length: residues per graph (used to reshape).
        bandwidth: scalar bandwidth σ_R(t).
    Returns:
        score: shape (N, 3) in the local tangent (so(3)) matching the model’s convention.
    """
    # Reshape into (B, L, 3, 3)
    B = R.shape[0] // sequence_length
    M = guiding_samples.shape[0] // sequence_length
    R_batched = R.view(B, sequence_length, 3, 3)
    G_batched = guiding_samples.view(M, sequence_length, 3, 3)

    # Pairwise relative rotations: Log(R^{-1} * G) -> right-trivialized rotvecs
    R_exp = R_batched[:, None]  # (B,1,L,3,3)
    G_exp = G_batched[None, ...]  # (1,M,L,3,3)
    rel = rot_mult(rot_transpose(R_exp), G_exp)  # (B,M,L,3,3)
    rotvec = rotmat_to_rotvec(rel)  # (B,M,L,3)

    # KDE weights
    sq_norm = (rotvec**2).sum(dim=-1)  # (B,M,L)
    density = torch.exp(-sq_norm / (2 * bandwidth**2))
    denom = density.sum(dim=1, keepdim=True) + eps
    weights = density / denom  # normalized over guiding samples M

    # Score: sum_i w_i * rotvec_i / sigma^2
    score = (weights[..., None] * rotvec).sum(dim=1) / (bandwidth**2)  # (B,L,3)
    return score.reshape(B * sequence_length, 3)


