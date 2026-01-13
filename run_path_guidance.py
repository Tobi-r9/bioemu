from bioemu.path_guidance import main as path_guidance
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gamma", type=float, default=0.1)
parser.add_argument("--N", type=int, default=100)
parser.add_argument("--n_calls", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=100)
args = parser.parse_args()

path_guidance(
    sequence="/lustre/groups/bauer/code/thoeppe/bioemu/protein_data/protein_check/protein_g.a3m",
    init_npz="/lustre/groups/bauer/code/thoeppe/bioemu/protein_data/transition_states/transition_states_positions_orientations.npz",
    N=args.N,
    eps_t=1e-3,
    max_t=0.98,
    method="euler",
    save_dir="/lustre/groups/bauer/code/thoeppe/bioemu/results/",
    gamma=args.gamma,
    batch_size=args.batch_size,
    n_calls=args.n_calls,
)
