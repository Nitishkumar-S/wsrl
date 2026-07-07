import collections
from typing import Optional

import minari
import numpy as np

from wsrl.envs.env_common import calc_return_to_go
from wsrl.utils.train_utils import concatenate_batches


def get_minari_dataset(
    dataset_id: str,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    clip_action: Optional[float] = None,
):
    """
    Drop-in replacement for get_d4rl_dataset() that loads data from a Minari
    dataset instead of D4RL.

    Iterates through all episodes in the Minari dataset and concatenates them
    into the flat dict format expected by the existing JAX ReplayBuffer:
        observations, actions, next_observations, rewards, masks, dones

    Args:
        dataset_id: Minari dataset identifier, e.g.
            "mujoco/halfcheetah/medium-replay-v0"
        reward_scale: Multiplier applied to each reward.
        reward_bias: Bias added to each reward after scaling.
        clip_action: If set, actions are clipped to [-clip_action, clip_action].

    Returns:
        A flat dict with keys:
            observations, actions, next_observations, rewards, masks, dones
        whose values are numpy arrays ready to be passed to subsample_batch().
    """
    dataset = minari.load_dataset(dataset_id, download=True)

    obs_list, next_obs_list, action_list = [], [], []
    reward_list, mask_list, done_list = [], [], []

    for episode in dataset.iterate_episodes():
        # Each episode has arrays: observations, actions, rewards,
        # terminations, truncations (all length T+1 for obs, T for the rest).
        T = len(episode.actions)
        ep_obs = np.array(episode.observations, dtype=np.float32)  # shape (T+1, obs_dim)
        ep_actions = np.array(episode.actions, dtype=np.float32)    # shape (T, act_dim)
        ep_rewards = np.array(episode.rewards, dtype=np.float32)    # shape (T,)
        ep_terminations = np.array(episode.terminations, dtype=np.float32)  # shape (T,)
        ep_truncations = np.array(episode.truncations, dtype=np.float32)    # shape (T,)

        # Consolidate termination and truncation into a single done flag,
        # matching the legacy d4rl convention used by the existing ReplayBuffer.
        ep_dones = np.logical_or(ep_terminations, ep_truncations).astype(np.float32)
        # mask=0 only on a true terminal (episode ended from environment logic),
        # consistent with the original get_d4rl_dataset behaviour.
        ep_masks = (1.0 - ep_terminations).astype(np.float32)

        obs_list.append(ep_obs[:T])        # current observations
        next_obs_list.append(ep_obs[1:])   # next observations
        action_list.append(ep_actions)
        reward_list.append(ep_rewards)
        mask_list.append(ep_masks)
        done_list.append(ep_dones)

    observations = np.concatenate(obs_list, axis=0)
    next_observations = np.concatenate(next_obs_list, axis=0)
    actions = np.concatenate(action_list, axis=0)
    rewards = np.concatenate(reward_list, axis=0)
    masks = np.concatenate(mask_list, axis=0)
    dones = np.concatenate(done_list, axis=0)

    if clip_action is not None:
        actions = np.clip(actions, -clip_action, clip_action)

    rewards = rewards * reward_scale + reward_bias

    return dict(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        masks=masks,
        dones=dones,
    )


def get_minari_dataset_with_mc_calculation(
    dataset_id: str,
    reward_scale: float,
    reward_bias: float,
    clip_action: Optional[float],
    gamma: float,
):
    """
    Minari equivalent for get_d4rl_dataset_with_mc_calculation.
    Calculates Monte Carlo returns per episode, required for Cal-QL.
    """
    dataset = minari.load_dataset(dataset_id, download=True)
    episodes_dict_list = []

    for episode in dataset.iterate_episodes():
        T = len(episode.actions)
        ep_obs = np.array(episode.observations, dtype=np.float32)  # shape (T+1, obs_dim)
        ep_actions = np.array(episode.actions, dtype=np.float32)   # shape (T, act_dim)
        ep_rewards = np.array(episode.rewards, dtype=np.float32)   # shape (T,)
        ep_terminations = np.array(episode.terminations, dtype=np.float32)  # shape (T,)
        ep_truncations = np.array(episode.truncations, dtype=np.float32)    # shape (T,)

        # Consolidate masks and dones
        ep_dones = np.logical_or(ep_terminations, ep_truncations).astype(np.float32)
        ep_masks = (1.0 - ep_terminations).astype(np.float32)

        # Apply reward scale and bias
        ep_rewards = ep_rewards * reward_scale + reward_bias

        # Apply action clipping if specified
        if clip_action is not None:
            ep_actions = np.clip(ep_actions, -clip_action, clip_action)

        # Calculate Monte Carlo return-to-go
        # Note: We pass dataset_id as the env_name string
        ep_mc_returns = calc_return_to_go(
            env_name=dataset_id,
            reward=ep_rewards,
            mask=ep_masks,
            gamma=gamma,
            reward_scale=reward_scale,
            reward_bias=reward_bias,
        )

        # Format into episode dict
        episode_data = dict(
            observations=ep_obs[:T],
            actions=ep_actions,
            next_observations=ep_obs[1:],
            rewards=ep_rewards,
            masks=ep_masks,
            dones=ep_dones,
            mc_returns=ep_mc_returns,
        )
        episodes_dict_list.append(episode_data)

    # Concatenate all episodes into a flat dictionary
    return concatenate_batches(episodes_dict_list)