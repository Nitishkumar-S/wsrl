from collections import defaultdict
from typing import Dict

import gym
import numpy as np


def flatten(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    policy_fn, env: gym.Env, num_episodes: int, clip_action: float = np.inf
) -> Dict[str, float]:
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            action = np.clip(action, -clip_action, clip_action)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats


def evaluate_with_trajectories(
    policy_fn, env: gym.Env, num_episodes: int, clip_action: float = np.inf
) -> Dict[str, float]:
    trajectories = []
    stats = defaultdict(list)

    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            action = np.clip(action, -clip_action, clip_action)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = dict(
                observations=observation,
                next_observations=next_observation,
                actions=action,
                rewards=r,
                dones=done,
                infos=info,
                masks=1 - terminated,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories
