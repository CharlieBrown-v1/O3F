import gym
import argparse

from pathlib import Path
from gym.envs.robotics import CollectEnv, ExecuteEnv, FetchEnv
from stable_baselines3 import HybridPPO, HybridSAC, HerReplayBuffer, HybridDQN
from stable_baselines3.common import torch_layers
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


default_cube_shape = [25, 35, 17]


def get_args():
    parser = argparse.ArgumentParser(description='Algorithm arguments')

    parser.add_argument('--id', type=str, default='Execute')
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_steps', type=int, default=2048)
    parser.add_argument('--kl', type=float, default=0.32)
    parser.add_argument('--name', type=str, default='ppo')
    parser.add_argument('--theta', type=float, default=0.02)
    parser.add_argument('--steps', type=int, default=50000000)

    args = parser.parse_args()

    return args


def make_env(env_name):
    def _thunk():
        env = gym.make(env_name)

        env = Monitor(env, None, allow_early_resets=True)

        return env

    return _thunk


def env_wrapper(env_name, num_envs):
    """
    :param env_name: 环境名称
    :param num_envs: 并行环境数量
    :return: 可并行采样的环境
    """
    envs = [
        make_env(env_name)
        for _ in range(num_envs)
    ]

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecNormalize(envs, norm_reward=True, norm_obs=False, training=False)

    return envs


from typing import Callable


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
    current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train_oe(train_id='Execute', num=10, 
                train_env: VecNormalize=None,
                train_steps=50000000,
                seed=7,
                target_kl=0.08,
                batch_size=128,
                n_steps=2048,
                theta=0.01,
                name='ppo'
                ):
    policy_name = 'HybridPolicy'
    kwargs = {
        'features_extractor_kwargs': {
            'cube_shape': default_cube_shape,
        },
    }
    fe_kwargs = {
    }
    print(f'fe_kwargs: {fe_kwargs}')
    kwargs = {
        'features_extractor_class': torch_layers.HybridExtractor,
        'features_extractor_kwargs': fe_kwargs,
    }
    if name == 'ppo':
        agent = HybridPPO(
            policy=policy_name,
            policy_kwargs=kwargs,
            env=train_env,
            batch_size=batch_size,
            n_steps=n_steps,
            seed=seed,
            target_kl=target_kl,
            verbose=1,
            tensorboard_log=f'oe_train',
            device=f'cuda:{get_best_cuda()}'
        )
        log_interval = 1
        save_interval = 1
    else:
        raise NotImplementedError

    save_path = f'./oe_model/batch_size_{batch_size}'
    save_path += f'_Nsteps_{n_steps}_kl_{target_kl}_{theta}'

    tb_log_name = f'batch_size_{batch_size}'
    tb_log_name += f'_Nsteps_{n_steps}_kl_{target_kl}_{theta}'

    agent.learn_one_step(
        total_timesteps=train_steps, log_interval=log_interval, save_interval=save_interval,
        save_path=save_path,
        tb_log_name=tb_log_name,
    )
    print('train finish!!!')


def get_best_cuda() -> int:
    import pynvml, numpy as np

    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    deviceMemory = []
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        deviceMemory.append(mem_info.free)
    deviceMemory = np.array(deviceMemory, dtype=np.int64)
    best_device_index = np.argmax(deviceMemory)
    print("best gpu:", best_device_index)
    return best_device_index


if __name__ == '__main__':
    args = get_args()
    assert isinstance(args.id, str), f'ID must be str while yours are {type(args.id)}'
    ID = args.id + 'Dense-v0'
    num   = args.num
    steps = args.steps
    seed  = args.seed
    batch_size  = args.batch_size
    n_steps  = args.n_steps
    kl    = args.kl
    theta = args.theta
    name  = args.name
    assert seed is not None
    
    env = env_wrapper(ID, num)
    
    train_oe(train_id=ID, num=num, train_env=env, train_steps=steps, seed=seed, target_kl=kl, n_steps=n_steps, batch_size=batch_size, name=name)
