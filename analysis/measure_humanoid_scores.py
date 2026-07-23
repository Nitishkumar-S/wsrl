"""
Measure the random / expert reference returns used to normalize Humanoid scores.

`get_locomotion_normalized_score()` in finetune.py needs a (random_score,
expert_score) pair per environment. halfcheetah/hopper/walker use the published
D4RL reference scores; Humanoid-v5 under Minari has no published pair, so it has
to be measured once.

Run this on the cluster, then paste the printed numbers into the "humanoid"
branch of get_locomotion_normalized_score() and delete the PROVISIONAL comment.

    python analysis/measure_humanoid_scores.py

The environment is recovered from the Minari dataset rather than built with
gym.make() so that it matches exactly what make_gym_env() constructs during
training (same env kwargs, same observation layout).
"""

import minari
import numpy as np

EXPERT_DATASET = "mujoco/humanoid/expert-v0"
N_RANDOM_EPISODES = 20
SEED = 0


def measure_random_score():
    """Mean episode return of a uniform-random policy."""
    dataset = minari.load_dataset(EXPERT_DATASET, download=True)
    env = dataset.recover_environment()
    env.action_space.seed(SEED)

    returns = []
    for episode in range(N_RANDOM_EPISODES):
        env.reset(seed=SEED + episode)
        total, done = 0.0, False
        while not done:
            _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
        returns.append(total)

    env.close()
    return float(np.mean(returns))


def measure_expert_score():
    """Mean episode return of the expert dataset's behaviour policy."""
    dataset = minari.load_dataset(EXPERT_DATASET, download=True)
    returns = [float(np.sum(ep.rewards)) for ep in dataset.iterate_episodes()]
    return float(np.mean(returns)), len(returns)


if __name__ == "__main__":
    random_score = measure_random_score()
    expert_score, n_episodes = measure_expert_score()

    print(f"random policy over {N_RANDOM_EPISODES} episodes")
    print(f"expert  {EXPERT_DATASET} over {n_episodes} episodes")
    print()
    print("paste into get_locomotion_normalized_score():")
    print(f"        random_score = {random_score:.6f}")
    print(f"        expert_score = {expert_score:.1f}")
