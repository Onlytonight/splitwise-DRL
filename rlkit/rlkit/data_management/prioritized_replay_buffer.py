from collections import OrderedDict

import numpy as np
import warnings

from rlkit.data_management.replay_buffer import ReplayBuffer


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer with importance sampling.

    This implementation is based on Schaul et al. (2016):
    "Prioritized Experience Replay".
    - Priorities are stored per transition.
    - Sampling probability: p_i = priority_i ** alpha / sum_j priority_j ** alpha
    - Importance sampling weight: w_i = (N * p_i) ** (-beta) / max_j w_j
    """

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        replace: bool = True,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 1e-3,
        eps: float = 1e-6,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros((max_replay_buffer_size, observation_dim))
        # Save next observations separately to avoid termination bookkeeping.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype="uint8")

        # env_infos
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = list(env_info_sizes.keys())

        # PER specific
        self._alpha = alpha
        self._beta = beta
        self._beta_increment_per_sampling = beta_increment_per_sampling
        self._eps = eps
        # Initialize all priorities to 1 so that they are sampled uniformly
        self._priorities = np.ones((max_replay_buffer_size,), dtype=np.float32)

        self._replace = replace
        self._top = 0
        self._size = 0

    def add_sample(
        self,
        observation,
        action,
        reward,
        next_observation,
        terminal,
        env_info,
        **kwargs,
    ):
        idx = self._top
        self._observations[idx] = observation
        self._actions[idx] = action
        self._rewards[idx] = reward
        self._terminals[idx] = terminal
        self._next_obs[idx] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][idx] = env_info.get(key, 0.0)

        # New samples get maximal priority so they are more likely to be seen
        if self._size > 0:
            max_priority = self._priorities[: self._size].max()
        else:
            max_priority = 1.0
        self._priorities[idx] = max_priority

        self._advance()

    def terminate_episode(self):
        pass

    def clear(self):
        self._top = 0
        self._size = 0
        # Keep priorities initialized to 1
        self._priorities.fill(1.0)

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def _sample_indices_and_weights(self, batch_size):
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")

        # Priorities only for filled part
        priorities = self._priorities[: self._size]
        # Avoid zero priorities
        priorities = np.maximum(priorities, self._eps)
        scaled_priorities = priorities ** self._alpha
        sample_prob = scaled_priorities / scaled_priorities.sum()

        replace = self._replace or self._size < batch_size
        if not self._replace and self._size < batch_size:
            warnings.warn(
                "Replace was set to false, but is temporarily set to true "
                "because batch size is larger than current size of replay."
            )

        indices = np.random.choice(
            self._size, size=batch_size, replace=replace, p=sample_prob
        )

        # Importance sampling weights
        self._beta = min(1.0, self._beta + self._beta_increment_per_sampling)
        weights = (self._size * sample_prob[indices]) ** (-self._beta)
        # Normalize weights so that max weight is 1
        weights /= weights.max()
        weights = weights.reshape(-1, 1).astype(np.float32)

        return indices, weights

    def random_batch(self, batch_size):
        indices, weights = self._sample_indices_and_weights(batch_size)

        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
            indices=indices.reshape(-1, 1),
            weights=weights,
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def update_priorities(self, indices, td_errors):
        """
        Update priorities given indices and corresponding TD errors.
        td_errors: 1D array-like of absolute TD errors.
        """
        indices = np.asarray(indices, dtype=np.int64).flatten()
        td_errors = np.asarray(td_errors, dtype=np.float32).flatten()
        # priority = |delta| + eps
        new_priorities = np.abs(td_errors) + self._eps
        self._priorities[indices] = new_priorities

    def rebuild_env_info_dict(self, idx):
        return {
            key: self._env_infos[key][idx]
            for key in self._env_info_keys
        }

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict(
            [
                ("size", self._size),
                ("max_priority", float(self._priorities[: self._size].max()) if self._size > 0 else 0.0),
                ("min_priority", float(self._priorities[: self._size].min()) if self._size > 0 else 0.0),
            ]
        )


