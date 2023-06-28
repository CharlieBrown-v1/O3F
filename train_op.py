import gym
import argparse

from pathlib import Path
from gym.envs.robotics import CollectEnv, PlanEnv
from stable_baselines3 import HybridPPO, HybridSAC, HerReplayBuffer, HybridDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize


default_cube_shape = [25, 35, 17]


def get_args():
    parser = argparse.ArgumentParser(description='Algorithm arguments')

    parser.add_argument('--id', type=str, default='Plan')
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--seed', type=int, default=47)
    parser.add_argument('--n_steps', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--kl', type=float, default=0.32)
    parser.add_argument('--name', type=str, default='ppo')
    parser.add_argument('--steps', type=int, default=1000000)

    args = parser.parse_args()

    return args


def make_env(env_name, agent_path, push_path=None, device=None):
    def _thunk():
        env = gym.make(env_name, agent_path=agent_path, device=device)
        env = Monitor(env, None, allow_early_resets=True)

        return env

    return _thunk


def env_wrapper(env_name, num_envs, agent_path=None, push_path=None, device=None):
    """
    :param env_name: 环境名称
    :param num_envs: 并行环境数量
    :return: 可并行采样的环境
    """
    envs = [
        make_env(env_name, agent_path, push_path, device)
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


def train_op(train_env, train_steps=250000, train_id='Plan',
                seed=7,
                n_steps=128,
                batch_size=64,
                target_kl=0.32,
                name='ppo',
                ):
    policy_name = 'HybridPolicy'
    from stable_baselines3.common.torch_layers import HybridUpperExtractor
    kwargs = {
        'features_extractor_class': HybridUpperExtractor,
        'features_extractor_kwargs': {'cube_shape': default_cube_shape},
    }
    seed = seed
    target_update_interval = 1000
    if name == 'ppo':
        agent = HybridPPO(
            policy=policy_name,
            env=train_env,
            batch_size=batch_size,
            n_steps=n_steps,
            seed=seed,
            target_kl=target_kl,
            policy_kwargs=kwargs,
            verbose=1,
            tensorboard_log=f'op_train',
            device=f'cuda:{get_best_cuda()}'
        )
        log_interval = 1
        save_interval = 1
    elif name == 'sac':
        agent = HybridSAC(
            policy=policy_name,
            env=train_env,
            seed=seed,
            batch_size=batch_size,
            buffer_size=30000,
            learning_starts=1000,
            verbose=1,
            tensorboard_log=f'op_train',
            device=f'cuda:{get_best_cuda()}'
        )
        log_interval = 10
        save_interval = 100
    elif name == 'dqn':
        agent = HybridDQN(
            policy=policy_name,
            env=train_env,
            seed=seed,
            batch_size=batch_size,
            buffer_size=50000,
            learning_starts=2500,
            target_update_interval=target_update_interval,
            verbose=1,
            tensorboard_log=f'op_train',
            device=f'cuda:{get_best_cuda()}'
        )
        log_interval = 10
        save_interval = 100
    else:
        raise NotImplementedError

    save_path = f'./op_model/{train_id[:-3]}_{name}_batch_size:{batch_size}'
    if name == 'ppo':
        save_path += f'_Nsteps:{n_steps}_kl:{target_kl}'
    else:
        save_path += f'_interval:{target_update_interval}'
    save_path += f'/{train_id[:-3]}_{name}'

    tb_log_name = f'{train_id[:-3]}_{name}_{batch_size}'
    if name == 'ppo':
        tb_log_name += f'_Nsteps:{n_steps}_kl:{target_kl}'
    else:
        tb_log_name += f'_interval:{target_update_interval}'

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
    ID = args.id + '-v0'
    num   = args.num
    steps = args.steps
    seed  = args.seed
    n_steps  = args.n_steps
    batch_size  = args.batch_size
    kl    = args.kl
    name  = args.name
    assert seed is not None
    
    if 'Plan' in ID:
        env = env_wrapper(ID, num, agent_path=f'./models/execute', device=f'cuda:{get_best_cuda()}')
    else:
        raise NotImplementedError
    
    train_op(train_env=env,
    train_steps=steps,
    train_id=ID, 
    seed=seed,
    n_steps=n_steps,
    batch_size=batch_size,
    target_kl=kl, 
    name=name,
    )
