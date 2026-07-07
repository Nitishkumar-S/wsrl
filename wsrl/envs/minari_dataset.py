import minari
import numpy as np
from typing import Optional

from wsrl.envs.env_common import calc_return_to_go
from wsrl.utils.train_utils import concatenate_batches

def get_minari_dataset(
    env_name: str,
    reward_scale: float = 1.0,
    reward_bias: float = 0.0,
    clip_action: Optional[float] = None,
):
    print(f"Loading Minari dataset: {env_name}")
    dataset = minari.load_dataset(env_name)

    observations, actions, next_observations, rewards, terminals = [], [], [], [], []
    
    for episode in dataset.iterate_episodes():
        observations.append(episode.observations[:-1])
        next_observations.append(episode.observations[1:])
        actions.append(episode.actions)
        rewards.append(episode.rewards)
        terminals.append(episode.terminations)
        
    # Concatenate all episodes into flat arrays
    dataset_dict = dict(
        observations=np.concatenate(observations, axis=0).astype(np.float32),
        actions=np.concatenate(actions, axis=0).astype(np.float32),
        next_observations=np.concatenate(next_observations, axis=0).astype(np.float32),
        rewards=np.concatenate(rewards, axis=0).astype(np.float32),
        terminals=np.concatenate(terminals, axis=0),
    )

    if clip_action:
        dataset_dict["actions"] = np.clip(dataset_dict["actions"], -clip_action, clip_action)

    dataset_dict["rewards"] = dataset_dict["rewards"] * reward_scale + reward_bias

    return dict(
        observations=dataset_dict["observations"],
        actions=dataset_dict["actions"],
        next_observations=dataset_dict["next_observations"],
        rewards=dataset_dict["rewards"],
        dones=dataset_dict["terminals"].astype(np.float32), 
        masks=1.0 - dataset_dict["terminals"].astype(np.float32), # Mask is 0 on terminal states for Q-bootstrapping
    )


def get_minari_dataset_with_mc_calculation(
    env_name: str,
    reward_scale: float,
    reward_bias: float,
    clip_action: Optional[float],
    gamma: float,
):
    print(f"Loading Minari dataset with MC returns: {env_name}")
    dataset = minari.load_dataset(env_name)

    episodes_dict_list = []
    
    for episode in dataset.iterate_episodes():
        obs = episode.observations[:-1].astype(np.float32)
        next_obs = episode.observations[1:].astype(np.float32)
        acts = episode.actions.astype(np.float32)
        rews = (episode.rewards * reward_scale + reward_bias).astype(np.float32)
        terms = episode.terminations
        
        if clip_action:
            acts = np.clip(acts, -clip_action, clip_action)
            
        mc_returns = calc_return_to_go(
            "halfcheetah", # Pass a generic locomotion string so it doesn't trigger sparse reward logic
            rews,
            1.0 - terms,
            gamma,
            reward_scale,
            reward_bias,
        )
        
        episode_data = dict(
            observations=obs,
            actions=acts,
            next_observations=next_obs,
            rewards=rews,
            terminals=terms,
            mc_returns=mc_returns,
        )
        episodes_dict_list.append(episode_data)
        
    concatenated = concatenate_batches(episodes_dict_list)
    
    return dict(
        observations=concatenated["observations"],
        actions=concatenated["actions"],
        next_observations=concatenated["next_observations"],
        rewards=concatenated["rewards"],
        dones=concatenated["terminals"].astype(np.float32),
        masks=1.0 - concatenated["terminals"].astype(np.float32),
        mc_returns=concatenated["mc_returns"],
    )