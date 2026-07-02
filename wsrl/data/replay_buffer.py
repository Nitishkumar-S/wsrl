import os
from typing import Iterable, Optional, Union

import gym
import gym.spaces
import numpy as np
from absl import flags

from wsrl.data.dataset import Dataset, DatasetDict, _sample
from wsrl.envs.env_common import calc_return_to_go


def _init_replay_dict(
    obs_space: gym.Space, capacity: int
) -> Union[np.ndarray, DatasetDict]:
    if isinstance(obs_space, gym.spaces.Box):
        return np.empty((capacity, *obs_space.shape), dtype=obs_space.dtype)
    elif isinstance(obs_space, gym.spaces.Dict):
        data_dict = {}
        for k, v in obs_space.spaces.items():
            data_dict[k] = _init_replay_dict(v, capacity)
        return data_dict
    else:
        raise TypeError()


def _insert_recursively(
    dataset_dict: DatasetDict, data_dict: DatasetDict, insert_index: int
):
    if isinstance(dataset_dict, np.ndarray):
        dataset_dict[insert_index] = data_dict
    elif isinstance(dataset_dict, dict):
        assert (
            dataset_dict.keys() == data_dict.keys()
        ), f"{dataset_dict.keys()} != {data_dict.keys()}"
        for k in dataset_dict.keys():
            _insert_recursively(dataset_dict[k], data_dict[k], insert_index)
    else:
        raise TypeError()


class ReplayBuffer(Dataset):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        if next_observation_space is None:
            next_observation_space = observation_space

        observation_data = _init_replay_dict(observation_space, capacity)
        next_observation_data = _init_replay_dict(next_observation_space, capacity)
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=bool),
            dones=np.empty((capacity,), dtype=np.float32),
        )

        super().__init__(dataset_dict, seed)

        self._size = 0
        self._capacity = capacity
        self._insert_index = 0
        self._sequential_index = 0
        self.unsampled_indices = list(range(self._size))
        self._discount = discount

    def __len__(self) -> int:
        return self._size

    def insert(self, data_dict: DatasetDict):
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        self._insert_index = (self._insert_index + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample_without_repeat(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
    ) -> dict:
        if keys is None:
            keys = self.dataset_dict.keys()

        batch = dict()
        if len(self.unsampled_indices) < batch_size:
            raise ValueError("Not enough samples left to sample without repeat.")
        selected_indices = []
        for _ in range(batch_size):
            idx = self.np_random.randint(len(self.unsampled_indices))
            selected_indices.append(self.unsampled_indices[idx])
            # Swap the selected index with the last unselected index
            self.unsampled_indices[idx], self.unsampled_indices[-1] = (
                self.unsampled_indices[-1],
                self.unsampled_indices[idx],
            )
            # Remove the last unselected index (which is now the selected index)
            self.unsampled_indices.pop()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], np.array(selected_indices))

        return batch

    def save(self, save_dir):
        save_buffer_file = os.path.join(save_dir, "online_buffer.npy")
        save_size_file = os.path.join(save_dir, "size.npy")
        np.save(save_buffer_file, self.dataset_dict)
        np.save(save_size_file, self._size)

    def load(self, save_dir):
        # TODO: maybe make sure the dataset_dict thats being loaded has mc_returns if self is ReplayBufferMC
        save_buffer_file = os.path.join(save_dir, "online_buffer.npy")
        save_size_file = os.path.join(save_dir, "size.npy")
        self.dataset_dict = np.load(save_buffer_file, allow_pickle=True).item()
        self._size = np.load(save_size_file, allow_pickle=True).item()
        self.unsampled_indices = list(range(self._size))


class ReplayBufferMC(ReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        next_observation_space: Optional[gym.Space] = None,
        seed: Optional[int] = None,
        discount: Optional[float] = None,
    ):
        assert discount is not None, "ReplayBufferMC requires a discount factor"
        super().__init__(
            observation_space,
            action_space,
            capacity,
            next_observation_space,
            seed,
            discount,
        )

        mc_returns = np.empty((capacity,), dtype=np.float32)
        self.dataset_dict["mc_returns"] = mc_returns

        self._allow_idxs = []
        self._traj_start_idx = 0

    def insert(self, data_dict: DatasetDict):
        # assumes replay buffer capacity is more than the number of online steps
        assert self._size < self._capacity, "replay buffer has reached capacity"

        data_dict["mc_returns"] = None
        _insert_recursively(self.dataset_dict, data_dict, self._insert_index)

        # if "dones" not in data_dict:
        #     data_dict["dones"] = 1 - data_dict["masks"]

        if data_dict["dones"] == 1.0:
            # compute the mc_returns
            FLAGS = flags.FLAGS
            rewards = self.dataset_dict["rewards"][
                self._traj_start_idx : self._insert_index + 1
            ]
            masks = self.dataset_dict["masks"][
                self._traj_start_idx : self._insert_index + 1
            ]
            self.dataset_dict["mc_returns"][
                self._traj_start_idx : self._insert_index + 1
            ] = calc_return_to_go(
                FLAGS.env,
                rewards,
                masks,
                self._discount,
            )

            self._allow_idxs.extend(
                list(range(self._traj_start_idx, self._insert_index + 1))
            )
            self._traj_start_idx = self._insert_index + 1

        self._size += 1
        self._insert_index += 1

    def sample(
        self,
        batch_size: int,
        keys: Optional[Iterable[str]] = None,
        indx: Optional[np.ndarray] = None,
    ) -> dict:
        if indx is None:
            indx = self.np_random.choice(
                self._allow_idxs, size=batch_size, replace=True
            )
        batch = dict()

        if keys is None:
            keys = self.dataset_dict.keys()

        for k in keys:
            batch[k] = _sample(self.dataset_dict[k], indx)

        return batch
