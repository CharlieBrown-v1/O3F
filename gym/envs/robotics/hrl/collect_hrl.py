import os
import copy
import numpy as np

from gym import utils
from gym.envs.robotics import fetch_env
from stable_baselines3 import HybridPPO


epsilon = 1e-2
desk_x = 0
desk_y = 1
pos_x = 2
pos_y = 3
pos_z = 4

action_list = [desk_x, desk_y, pos_x, pos_y, pos_z]
MODEL_XML_PATH = os.path.join("hrl", "collect_hrl.xml")


def vector_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class CollectExecuteEnv(fetch_env.FetchEnv, utils.EzPickle):
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
            block_gripper=False,
            n_substeps=20,
            gripper_extra_height=0.2,
            target_in_the_air=True,
            target_offset=0.0,
            obj_range=0.15,
            target_range=0.15,
            distance_threshold=0.05,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            single_count_sup=5,
            target_in_air_probability=0.5,
            object_stacked_probability=0.5,
            hrl_mode=True,
            # random_mode=True,
            collect_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

        self.training_mode = True

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

        self.reward_factor = 0.1
        self.prev_valid_count = None

        self.desired_boundary_list = [
            np.array([1.30 - 0.16 + 0.025, 0.75 - 0.16 + 0.025]),
            np.array([1.30 - 0.16 + 0.025, 0.75 + 0.16 - 0.025]),
            np.array([1.30 + 0.16 - 0.025, 0.75 - 0.16 + 0.025]),
            np.array([1.30 + 0.16 - 0.025, 0.75 + 0.16 - 0.025]),
        ]
        self.lower_bound = np.array([1.30 - 0.16 + 0.025, 0.75 - 0.16 + 0.025])
        self.upper_bound = np.array([1.30 + 0.16 - 0.025, 0.75 + 0.16 - 0.025])

        self.count = 0
        
    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset_indicate(self):
        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

    def reset_after_removal(self, goal=None):
        assert self.hrl_mode
        assert self.is_removal_success

        if goal is None:
            goal = self.global_goal.copy()

        new_achieved_name = 'target_object'
        new_obstacle_name_list = self.object_name_list.copy()
        new_obstacle_name_list.remove(new_achieved_name)

        self.achieved_name = copy.deepcopy(new_achieved_name)
        self.obstacle_name_list = new_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.get_xpos(obstacle_name).copy() for obstacle_name
                                        in self.obstacle_name_list]

        self._state_init(goal.copy())

    def macro_step_setup(self, macro_action):
        removal_goal = np.array([macro_action[desk_x], macro_action[desk_y], self.height_offset])
        action_xpos = np.array([macro_action[pos_x], macro_action[pos_y], macro_action[pos_z]])

        achieved_name = None
        min_dist = np.inf
        name_list = self.object_name_list
        for name in name_list:
            xpos = self.get_xpos(name).copy()
            dist = vector_distance(action_xpos, xpos)
            if dist < min_dist:
                min_dist = dist
                achieved_name = name
        assert achieved_name is not None

        # np.save(f'/home/stalin/robot/result/collect/{self.count}', removal_goal)
        self.count += 1

        self.achieved_name = achieved_name
        self.removal_goal = removal_goal.copy()
        self.achieved_name_indicate = achieved_name
        self.removal_goal_indicate = removal_goal.copy()
        self.removal_xpos_indicate = action_xpos.copy()

        tmp_obstacle_name_list = self.object_name_list.copy()
        tmp_obstacle_name_list.remove(self.achieved_name)
        self.obstacle_name_list = tmp_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.get_xpos(name).copy() for name in self.obstacle_name_list]

        return achieved_name, removal_goal, min_dist

    def macro_step(self, agent: HybridPPO, obs: dict):
        i = 0
        info = {'is_success': False}
        frames = []
        while i < self.spec.max_episode_steps:
            i += 1
            agent_action = agent.predict(observation=obs, deterministic=True)[0]
            next_obs, reward, done, info = self.step(agent_action)
            obs = next_obs
            # frames.append(self.render(mode='rgb_array'))
            if self.training_mode:
                self.sim.forward()
            else:
                self.render()
            if info['train_done']:
                break
        info['frames'] = frames
        reward = self.collect_compute_reward(achieved_goal=None, goal=None, info=info)
        if info['is_removal_success']:
            return obs, reward, False, info
        else:
            return obs, reward, True, info

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

    def get_obs(self, achieved_name=None, goal=None):
        assert self.hrl_mode

        new_goal = goal.copy() if goal is not None else None
        self.is_grasp = False
        self.is_removal_success = False

        if achieved_name is not None:
            self.achieved_name = copy.deepcopy(achieved_name)
        else:
            self.achieved_name = 'target_object'

        if new_goal is not None and np.any(new_goal != self.global_goal):
            self.removal_goal = new_goal.copy()
        else:
            self.removal_goal = None
            new_goal = self.global_goal.copy()

        tmp_obstacle_name_list = self.object_name_list.copy()
        tmp_obstacle_name_list.remove(self.achieved_name)
        self.obstacle_name_list = tmp_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.get_xpos(name).copy() for name in self.obstacle_name_list]

        self._state_init(new_goal.copy())
        return self._get_obs()

    def counting_valid_object(self):
        count = 0
        for name in self.object_name_list:
            object_xy = self.get_xpos(name)[:2].copy()
            if np.all(object_xy >= self.lower_bound) and np.all(object_xy <= self.upper_bound):
                count += 1
        return count

    def reset(self):
        self.prev_valid_count = 0
        self.count = 0
        return super(CollectExecuteEnv, self).reset()

    def collect_compute_reward(self, achieved_goal, goal, info):
        prev_valid_count = self.prev_valid_count
        curr_valid_count = self.counting_valid_object()
        self.prev_valid_count = curr_valid_count
        return self.reward_factor * (curr_valid_count - prev_valid_count)

    def is_collect_success(self):
        valid_count = self.counting_valid_object()
        return valid_count == len(self.object_name_list)

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        global_target_site_id = self.sim.model.site_name2id("global_target")
        area_id = self.sim.model.site_name2id("area")
        removal_target_site_id = self.sim.model.site_name2id("removal_target")
        achieved_site_id = self.sim.model.site_name2id("achieved_site")

        if self.removal_goal_indicate is not None:
            self.sim.model.site_pos[global_target_site_id] = self.removal_goal_indicate - sites_offset[global_target_site_id]
        elif self.removal_goal is not None:
            self.sim.model.site_pos[global_target_site_id] = self.removal_goal - sites_offset[global_target_site_id]
        else:
            self.sim.model.site_pos[global_target_site_id] = np.array([20, 20, 0.5])

        self.sim.model.site_pos[global_target_site_id] = np.array([20, 20, 0.5])
        self.sim.model.site_pos[area_id] = np.array([1.30, 0.75, 0.4 - 0.01 + 1e-5])  - sites_offset[area_id]

        if self.removal_goal_indicate is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal_indicate - sites_offset[
                removal_target_site_id]
        elif self.removal_goal is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[
                removal_target_site_id]
        else:
            self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])

        if self.achieved_name_indicate is not None:
            self.sim.model.site_pos[achieved_site_id] = self.get_xpos(
                self.achieved_name_indicate).copy() - sites_offset[achieved_site_id]
        else:
            self.sim.model.site_pos[achieved_site_id] = self.get_xpos(name=self.achieved_name).copy() - \
                                                        sites_offset[achieved_site_id]

        self.sim.forward()
