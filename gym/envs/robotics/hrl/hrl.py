import os
import copy
import numpy as np

from gym import utils, spaces
from typing import Tuple
from gym.envs.robotics import fetch_env

MODEL_XML_PATH = os.path.join("hrl", "hrl.xml")

epsilon = 1e-3
desk_x = 0
desk_y = 1
desk_z = 2
pos_x = 3
pos_y = 4
pos_z = 5
action_list = [desk_x, desk_y, desk_z, pos_x, pos_y, pos_z]


def vector_distance(goal_a, goal_b, dist_sup=None):
    assert goal_a.shape == goal_b.shape
    if dist_sup is None:
        dist_sup = np.inf
    return min(np.linalg.norm(goal_a - goal_b, axis=-1), dist_sup)


class ExecuteEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, distance_threshold=0.02, reward_type='dense'):
        initial_qpos = {
            "robot0:slide0": 0.405,
            "robot0:slide1": 0.48,
            "robot0:slide2": 0.0,
        }

        self.training_mode = True

        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

        self.finished_count = None
        self.lower_reward_sup = 0.12
        self.valid_dist_sup = 0.3

        step_size = 0.05
        self.object_size = 0.05
        self.obstacle_goal_0 = np.array([1.30, 0.65, 0.425 + 0 * step_size])
        self.obstacle_goal_1 = np.array([1.30, 0.65, 0.425 + 1 * step_size])
        self.target_goal = np.array([1.30, 0.65, 0.425 + 2 * step_size])
        self.final_goal = np.array([1.30, 0.75, 0.425 + 4 * step_size])

        table_xy = np.array([1.3, 0.75])
        table_size = np.array([0.25, 0.35])
        table_start_xy = table_xy - table_size + step_size
        table_end_xy = table_xy + table_size - step_size
        table_start_z = 0.425
        table_end_z = 0.425 + 0.3
        self.table_start_xyz = np.r_[table_start_xy, table_start_z]
        self.table_end_xyz = np.r_[table_end_xy, table_end_z]
        self.upper_action_space = spaces.Box(-1, 1, shape=(2 * self.table_start_xyz.shape[0],))
        self.goal_list = [
            self.obstacle_goal_0.copy(),
            self.obstacle_goal_1.copy(),
            self.target_goal.copy(),
            self.final_goal.copy(),
        ]
        self.name_list = [
            'obstacle_object_0',
            'obstacle_object_1',
            'target_object',
            'robot0:grip',
        ]
        assert len(self.goal_list) == len(self.name_list)
        self.count2goal = dict(zip(range(len(self.goal_list)), self.goal_list))
        self.count2name = dict(zip(range(len(self.name_list)), self.name_list))

        self.task_list = [
            'stack',
            'dismantle',
            'random',
        ]

        self.task = 'random'  # indicate current task
        self.init_xpos = None  # used by task != random
        self.free_object_name_list = None
        self.fixed_object_name_list = None

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
            distance_threshold=distance_threshold,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            single_count_sup=7,
            target_in_air_probability=0.5,
            hrl_mode=True,
            random_mode=True,
            stack_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def set_mode(self, name: str, mode: bool):
        if name == 'training':
            self.training_mode = mode
        else:
            raise NotImplementedError

    def reset(self):
        self.task = np.random.choice(self.task_list)
        self.free_object_name_list = []
        self.fixed_object_name_list = []
        obs = super(ExecuteEnv, self).reset()

        return obs

    def reset_indicate(self):
        self.achieved_name_indicate = None
        self.removal_goal_indicate = None
        self.removal_xpos_indicate = None

    def action2xpos(self, action: np.ndarray) -> np.ndarray:
        planning_action = action.copy()

        # action for choosing desk's position
        planning_action[:3] = (self.table_end_xyz[:3] - self.table_start_xyz[:3]) * planning_action[:3] / 2 \
                              + (self.table_start_xyz[:3] + self.table_end_xyz[:3]) / 2
        # action for choosing obstacle's position
        planning_action[3:] = (self.table_end_xyz - self.table_start_xyz) * planning_action[3:] / 2 \
                              + (self.table_start_xyz + self.table_end_xyz) / 2
        return planning_action

    def reset_after_removal(self, goal=None, info=None):
        assert self.hrl_mode

        assert info['is_removal_success']
        self.free_object_name_list.remove(self.achieved_name)
        if self.task == 'stack':
            new_achieved_name = np.random.choice(self.free_object_name_list)
        elif self.task == 'dismantle':
            new_achieved_name = self.get_dismantle_achieved_name(self.free_object_name_list)
        elif self.task == 'random':
            new_achieved_name = np.random.choice(self.free_object_name_list)
        else:
            raise NotImplementedError

        obs = self._get_obs()
        if self.task == 'stack':
            new_removal_goal = self.init_xpos + np.array([0, 0, self.finished_count * self.object_size])
        elif self.task in ['dismantle', 'random']:
            new_removal_action = self.obs2goal(obs=obs)
            macro_action = self.action2xpos(action=new_removal_action)
            new_removal_goal = macro_action[:3]
            new_removal_goal[2] = self.table_start_xyz[2]
            if self.finished_count == len(self.object_name_list) - 1:
                new_removal_goal[2] = np.random.uniform(self.table_start_xyz[2], self.table_end_xyz[2])
        else:
            raise NotImplementedError

        self.removal_goal = new_removal_goal.copy()
        new_obstacle_name_list = self.object_name_list.copy()

        try:
            new_obstacle_name_list.remove(new_achieved_name)
        except ValueError:
            print(f'Bullfuck!')
            pass

        self.achieved_name = copy.deepcopy(new_achieved_name)
        self.obstacle_name_list = new_obstacle_name_list.copy()
        self.init_obstacle_xpos_list = [self.get_xpos(obstacle_name).copy() for obstacle_name
                                        in self.obstacle_name_list]

        self._state_init(new_removal_goal.copy())

    def hrl_step(self, obs, action):
        info = {
            "is_grasp": False,
            "is_removal_success": False,
            "is_success": False,
            "is_fail": self._is_fail(),
        }

        # DIY
        if not self.is_grasp:
            self.is_grasp = self._judge_is_grasp(obs['achieved_goal'].copy(), action.copy())

            if self.is_grasp:
                info['is_grasp'] = True

        # DIY
        if self.removal_goal is not None and not self.is_removal_success:
            is_removal_success = self._is_success(obs["achieved_goal"], self.removal_goal)

            if is_removal_success:
                self.removal_goal = None
                self.finished_count += 1
                info['is_removal_success'] = True

        # DIY
        removal_done = self.removal_goal is None or self.is_removal_success
        # done for reset sim
        info['is_success'] = self.finished_count >= len(self.object_name_list)

        done = info['is_fail'] or info['is_success']
        # train_* for train a new trial
        info['train_done'] = info['is_fail'] or info['is_success'] or info['is_removal_success']
        info['train_is_success'] = info['is_success'] or info['is_removal_success']

        # DIY
        info['removal_done'] = (not removal_done and info['is_fail']) or info['is_removal_success']
        info['removal_success'] = info['is_removal_success']
        info['global_done'] = (removal_done and info['is_fail']) or info['is_success']
        info['global_success'] = info['is_success']

        # DIY
        if self.removal_goal is None or self.is_removal_success:
            goal = self.global_goal.copy()
        else:
            goal = self.removal_goal.copy()

        reward = self.compute_reward(obs["achieved_goal"], goal, info)

        if info['is_removal_success'] and not info['is_success']:
            self.reset_after_removal(info=info.copy())

        return obs, reward, done, info

    def action2feature(self, macro_action: np.ndarray) -> Tuple[str, np.ndarray, float]:
        removal_goal = np.array([macro_action[desk_x], macro_action[desk_y], macro_action[desk_z]])
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

        return achieved_name, removal_goal, min_dist

    def macro_step_setup(self, macro_action):
        achieved_name, removal_goal, min_dist = self.action2feature(macro_action=macro_action)
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

    def judge(self, name_list: list, xpos_list: list, mode: str):
        assert len(name_list) == len(xpos_list)

        achieved_xpos = self.get_xpos(name=self.achieved_name).copy()

        move_count = 0
        not_in_desk_count = int(achieved_xpos[2] <= 0.4 - 0.01)

        for idx in np.arange(len(name_list)):
            name = name_list[idx]
            init_xpos = np.array(xpos_list[idx].copy())
            curr_xpos = self.get_xpos(name=name).copy()
            delta_xpos = vector_distance(init_xpos, curr_xpos)

            if delta_xpos > 0.05:
                move_count += 1

            if curr_xpos[2] <= 0.4 - 0.01:
                not_in_desk_count += 1

        if mode == 'done':
            return move_count + not_in_desk_count > 0
        elif mode == 'punish':
            return (move_count + not_in_desk_count) * self.punish_factor

    def reset_removal(self, goal: np.ndarray, removal_goal=None, is_removal=True):
        self.is_grasp = False
        self.is_removal_success = False
        self.removal_goal = removal_goal.copy()
        self._state_init(self.removal_goal.copy())

    def obs2goal(self, obs) -> np.ndarray:
        goal = self.upper_action_space.sample()

        return goal

    def get_dismantle_achieved_name(self, name_list: list) -> str:
        achieved_name = None
        highest_height = -np.inf
        for name in name_list:
            xpos = self.get_xpos(name).copy()
            if xpos[2] > highest_height:
                achieved_name = name
                highest_height = xpos[2]
        assert achieved_name is not None

        return achieved_name

    def _sample_goal(self):
        # global goal is useless
        obs = self._get_obs()
        global_action = self.obs2goal(obs=obs)
        macro_action = self.action2xpos(action=global_action)
        init_xpos = macro_action[:3]
        init_xpos[2] = self.table_start_xyz[2]  # set the first object above desk exactly
        self.init_xpos = init_xpos.copy()
        deterministic_count = np.random.randint(len(self.object_name_list))
        if self.task == 'stack':
            fixed_count = deterministic_count
            free_count = len(self.object_name_list) - fixed_count
            self.fixed_object_name_list = list(np.random.choice(self.object_name_list, fixed_count, replace=False))
            self.free_object_name_list = [object_name for object_name in self.object_name_list
                                          if object_name not in self.fixed_object_name_list]
            deterministic_name_list = self.fixed_object_name_list.copy()
        elif self.task == 'dismantle':
            deterministic_count += 1  # [0, n-1] -> [1, n]
            free_count = deterministic_count
            fixed_count = len(self.object_name_list) - free_count
            self.free_object_name_list = list(np.random.choice(self.object_name_list, free_count, replace=False))
            self.fixed_object_name_list = [object_name for object_name in self.object_name_list
                                          if object_name not in self.free_object_name_list]
            deterministic_name_list = self.free_object_name_list.copy()
        elif self.task == 'random':
            deterministic_count = 0
            fixed_count = deterministic_count
            free_count = len(self.object_name_list) - fixed_count
            self.fixed_object_name_list = []
            self.free_object_name_list = self.object_name_list.copy()
            deterministic_name_list = self.fixed_object_name_list.copy()
        else:
            raise NotImplementedError
        assert fixed_count + free_count == len(self.object_name_list)

        # print(f'count: {deterministic_count}')
        # print(f'list: {deterministic_name_list}')
        # print(f'task: {self.task}')
        for idx in range(deterministic_count):
            deterministic_name = deterministic_name_list[idx]
            deterministic_xpos = self.init_xpos + np.array([0, 0, idx * self.object_size])
            self.sim.data.set_joint_qpos(f'{deterministic_name}:joint', np.r_[deterministic_xpos, self.object_generator.qpos_postfix])
        self.sim.forward()
        for _ in range(10):
            self.sim.step()
        macro_action = np.zeros(len(action_list))  # ignore global goal in reset scene
        new_achieved_name, goal, min_dist = self.action2feature(macro_action=macro_action)

        if self.task == 'stack':
            removal_goal = self.init_xpos + np.array([0, 0, fixed_count * self.object_size])
            new_achieved_name = np.random.choice(self.free_object_name_list)
            achieved_xpos = self.get_xpos(new_achieved_name).copy()
        elif self.task == 'dismantle':
            removal_action = self.obs2goal(obs=obs)
            macro_action = self.action2xpos(action=removal_action)
            removal_goal = macro_action[:3]
            removal_goal[2] = self.table_start_xyz[2]
            new_achieved_name = self.get_dismantle_achieved_name(self.free_object_name_list)
            achieved_xpos = self.get_xpos(new_achieved_name).copy()
        elif self.task == 'random':
            removal_action = self.obs2goal(obs=obs)
            macro_action = self.action2xpos(action=removal_action)
            removal_goal = macro_action[:3]
            removal_goal[2] = self.table_start_xyz[2]
            new_achieved_name = np.random.choice(self.free_object_name_list)
            achieved_xpos = self.get_xpos(new_achieved_name).copy()
        else:
            raise NotImplementedError

        # use finished_count to guide reset of sim
        self.finished_count = fixed_count
        self.reset_removal(goal=goal.copy(), removal_goal=removal_goal.copy())
        self.macro_step_setup(macro_action=np.r_[
            removal_goal.copy(),
            achieved_xpos.copy(),
        ])

        return goal.copy()

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        global_target_site_id = self.sim.model.site_name2id("global_target")
        removal_target_site_id = self.sim.model.site_name2id("removal_target")

        final_goal_id = self.sim.model.site_name2id("final_goal")
        target_goal_id = self.sim.model.site_name2id("target_goal")
        obstacle_goal_0_id = self.sim.model.site_name2id("obstacle_goal_0")
        obstacle_goal_1_id = self.sim.model.site_name2id("obstacle_goal_1")

        # self.sim.model.site_pos[global_target_site_id] = self.global_goal - sites_offset[global_target_site_id]
        if self.removal_goal is not None:
            self.sim.model.site_pos[removal_target_site_id] = self.removal_goal - sites_offset[removal_target_site_id]
        else:
            self.sim.model.site_pos[removal_target_site_id] = np.array([20, 20, 0.5])

        # self.sim.model.site_pos[final_goal_id] = self.final_goal - sites_offset[final_goal_id]
        # self.sim.model.site_pos[target_goal_id] = self.target_goal - sites_offset[target_goal_id]
        # self.sim.model.site_pos[obstacle_goal_0_id] = self.obstacle_goal_0 - sites_offset[obstacle_goal_0_id]
        # self.sim.model.site_pos[obstacle_goal_1_id] = self.obstacle_goal_1 - sites_offset[obstacle_goal_1_id]

        self.sim.forward()
