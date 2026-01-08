import torch

from bioemu.mew_utils import so3_gaussian_kde_score_batched


def test_so3_kde_score_identity_zero():
    seq_len = 2
    B = 1
    R = torch.eye(3).repeat(B * seq_len, 1, 1)
    guiding = R.clone()
    score = so3_gaussian_kde_score_batched(
        R=R,
        guiding_samples=guiding,
        sequence_length=seq_len,
        bandwidth=0.5,
    )
    assert score.shape == (B * seq_len, 3)
    assert torch.allclose(score, torch.zeros_like(score), atol=1e-6)

