import os
import copy
import numpy as np

from gym import utils
from gym.envs.robotics import fetch_env
from stable_baselines3 import HybridPPO


desk_x = 0
desk_y = 1
pos_x = 2
pos_y = 3
pos_z = 4

action_list = [desk_x, desk_y, pos_x, pos_y, pos_z]
MODEL_XML_PATH = os.path.join("hrl", "render_hrl.xml")


def vector_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class TestExecuteEnv(fetch_env.FetchEnv, utils.EzPickle):
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
            single_count_sup=18,
            hrl_mode=True,
            # random_mode=True,  # True for testing, False for fine-tuning
            test_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

        self.test_mode = False
        self.planning_mode = False
        self.training_mode = True

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

    def set_mode(self, name: str, mode: bool):
        if name == 'test':
            self.test_mode = mode
        elif name == 'planning':
            self.planning_mode = mode
        elif name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset(self):
        obs = super(TestExecuteEnv, self).reset()
        self.release_flag = False
        self.count = 0

        return obs

    def reset_indicate(self):
        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

    def reset_after_removal(self, goal=None):
        assert self.hrl_mode
        assert self.is_removal_success

        if goal is None:
            goal = self.global_goal.copy()

        new_achieved_name = self.object_generator.global_achieved_name
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

        if achieved_name != 'target_object' and np.any(removal_goal != self.global_goal):
            self.achieved_name = achieved_name
            self.removal_goal = removal_goal.copy()
            self.achieved_name_indicate = achieved_name
            self.removal_goal_indicate = removal_goal.copy()
            self.removal_xpos_indicate = action_xpos.copy()
        else:
            self.achieved_name = self.object_generator.global_achieved_name
            self.removal_goal = None
            self.reset_indicate()
            self.removal_xpos_indicate = action_xpos.copy()
            achieved_name = None
            removal_goal = None
            min_dist = None

        tmp_obstacle_name_list = self.object_name_list.copy()
        tmp_obstacle_name_list.remove(self.achieved_name)
        self.obstacle_name_list = tmp_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.get_xpos(name).copy() for name in self.obstacle_name_list]

        return achieved_name, removal_goal, min_dist

    def macro_step(self, agent: HybridPPO, obs: dict):
        i = 0
        info = {'is_success': False}
        frames = []

        probability = np.random.uniform()
        while i < self.spec.max_episode_steps:
            i += 1
            agent_action = agent.predict(observation=obs, deterministic=True)[0]
            if probability < 0.5:
                if vector_distance(self.get_xpos(self.achieved_name), self.get_xpos('robot0:grip')) < self.distance_threshold:
                    self.count += 1
                if self.count > 3:
                    agent_action[-1] = 1
                    probability = 1
            next_obs, reward, done, info = self.step(agent_action)
            obs = next_obs
            # frames.append(self.render(mode='rgb_array'))
            # if self.training_mode or self.planning_mode or self.test_mode:
            if self.training_mode or self.planning_mode:
                self.sim.forward()
            else:
                self.render()
            if info['train_done']:
                break
        info['frames'] = frames
        if info['is_removal_success']:
            return obs, 0, False, info
        else:
            return obs, 0, True, info

    def is_fail(self):
        return self.judge(self.obstacle_name_list.copy(), self.init_obstacle_xpos_list.copy(), mode='done')

    def get_obs(self, achieved_name=None, goal=None):
        assert self.hrl_mode

        new_goal = goal.copy() if goal is not None else None
        self.is_grasp = False
        self.is_removal_success = False

        if achieved_name is not None:
            self.achieved_name = copy.deepcopy(achieved_name)
        else:
            self.achieved_name = self.object_generator.global_achieved_name

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

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        global_target_site_id = self.sim.model.site_name2id("global_target")
        removal_target_site_id = self.sim.model.site_name2id("removal_target")
        removal_indicate_site_id = self.sim.model.site_name2id("removal_indicate")
        achieved_site_id = self.sim.model.site_name2id("achieved_site")
        cube_site_id = self.sim.model.site_name2id("cube_site")

        if self.planning_mode:
            self.sim.model.site_pos[global_target_site_id] = self.global_goal - sites_offset[global_target_site_id]

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
        elif not self.test_mode:
            self.sim.model.site_pos[global_target_site_id] = self.global_goal - sites_offset[global_target_site_id]

            if self.removal_goal_indicate is not None:
                self.sim.model.site_pos[removal_target_site_id] = self.removal_goal_indicate - sites_offset[
                    removal_target_site_id]
            elif self.removal_goal is not None:
                self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[removal_target_site_id]
            else:
                self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])

            if self.removal_xpos_indicate is not None:
                self.sim.model.site_pos[removal_indicate_site_id] = self.removal_xpos_indicate - sites_offset[
                    removal_indicate_site_id]
            else:
                self.sim.model.site_pos[removal_indicate_site_id] = np.array([20, 20, 0.5])

            if self.achieved_name_indicate is not None:
                self.sim.model.site_pos[achieved_site_id] = self.get_xpos(
                    self.achieved_name_indicate).copy() - sites_offset[achieved_site_id]
            else:
                self.sim.model.site_pos[achieved_site_id] = self.get_xpos(name=self.achieved_name).copy() - \
                                                            sites_offset[achieved_site_id]
            self.sim.model.site_pos[cube_site_id] = self.cube_starting_point.copy() - sites_offset[cube_site_id]
        else:
            self.sim.model.site_pos[achieved_site_id] = np.array([30, 30, 0.5])

        self.sim.forward()
