import numpy as np
from baselines.common.vec_env.subproc_vec_env import  SubprocVecEnv


class MultiModalSubprocVecEnv(SubprocVecEnv):
    """
        In principle the  same as SubprocVecEnv, but adapted for multi modal environments
        like our Score Following Environment
    """

    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        super().__init__(env_fns=env_fns)

    def step_wait(self):

        results = [remote.recv() for remote in self.remotes]
        self.waiting = False

        obs, rews, dones, infos = zip(*results)

        observations = [np.stack([observation[i] for observation in obs])
                for i in range(len(self.observation_space.spaces))]

        return observations, np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))

        remotes_list = [remote.recv() for remote in self.remotes]

        return [np.stack([remote[i] for remote in remotes_list])
                for i in range(len(self.observation_space.spaces))]
