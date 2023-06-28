import os
import numpy as np
from gym import utils
from gym import spaces
from gym.envs.robotics import fetch_env
from gym.envs.robotics import utils as robot_utils


MODEL_XML_PATH = os.path.join("hrl", "hrl.xml")


def vector_distance(goal_a: np.ndarray, goal_b: np.ndarray):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PushEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type="dense"):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self,
            MODEL_XML_PATH,
            has_object=True,
            block_gripper=True,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=False,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            single_count_sup=7,
            target_in_air_probability=0,
            object_stacked_probability=0,
            hrl_mode=True,
            random_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def _sample_goal(self):
        is_removal = False

        assert self.has_object
        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3
        )
        goal += self.target_offset

        achieved_xpos = self.get_xpos(name=self.achieved_name).copy()
        goal[2] = achieved_xpos[2]

        self.reset_removal(goal=goal.copy(), is_removal=is_removal)

        return goal.copy()

    def hrl_reward(self, achieved_goal, goal, info):
        assert self.reward_type == 'dense'

        grip_pos = self.get_xpos("robot0:grip").copy()

        curr_grip_achi_dist = vector_distance(np.broadcast_to(grip_pos, achieved_goal.shape), achieved_goal)
        grip_achi_reward = self.prev_grip_achi_dist - curr_grip_achi_dist
        self.prev_grip_achi_dist = curr_grip_achi_dist

        curr_achi_desi_dist = vector_distance(achieved_goal, goal)
        achi_desi_reward = self.prev_achi_desi_dist - curr_achi_desi_dist
        self.prev_achi_desi_dist = curr_achi_desi_dist

        reward = self.reward_factor * (grip_achi_reward + achi_desi_reward)

        is_success = info['train_is_success']
        reward = np.where(1 - is_success, reward, self.success_reward)
        reward += self.judge(self.obstacle_name_list.copy(), self.init_obstacle_xpos_list.copy(), mode='punish')

        return reward

    def judge(self, name_list: list, xpos_list: list, mode: str):
        assert len(name_list) == len(xpos_list)

        achieved_xpos = self.get_xpos(name=self.achieved_name).copy()

        not_in_desk_count = int(achieved_xpos[2] <= 0.4 - 0.01)

        for idx in np.arange(len(name_list)):
            name = name_list[idx]
            curr_xpos = self.get_xpos(name).copy()

            if curr_xpos[2] <= 0.4 - 0.01:
                not_in_desk_count += 1

        if mode == 'done':
            return not_in_desk_count > 0
        elif mode == 'punish':
            return not_in_desk_count * self.punish_factor
