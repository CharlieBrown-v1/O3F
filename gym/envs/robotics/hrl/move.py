import os
import numpy as np
from gym import utils
from gym.envs.robotics import fetch_env, rotations
from gym.envs.robotics import utils as robot_utils


epsilon = 1e-3


gripper_finger_size = np.array([0.0135, 0.0070, 0.0385])
gripper_link_size = np.array([0.0403, 0.0631, 0.0617])
length_scale = 25
width_scale = 35
height_scale = 17


MODEL_XML_PATH = os.path.join("hrl", "hrl.xml")


def vector_distance(goal_a: np.ndarray, goal_b: np.ndarray):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class MoveEnv(fetch_env.FetchEnv, utils.EzPickle):
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
            distance_threshold=0.02,
            initial_qpos=initial_qpos,
            reward_type=reward_type,
            single_count_sup=7,
            object_stacked_probability=0,
            hrl_mode=True,
            random_mode=True,
        )
        utils.EzPickle.__init__(self, reward_type=reward_type)

    def _sample_goal(self):
        is_removal = False

        goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(
            -self.target_range, self.target_range, size=3
        )
        goal += self.target_offset

        self.reset_removal(goal=goal.copy(), is_removal=is_removal)

        return goal.copy()

    def _get_obs(self):
        # positions
        grip_pos = self.get_xpos("robot0:grip").copy()
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp("robot0:grip").copy() * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)

        gripper_state = robot_qpos[-2:]
        gripper_vel = (
                robot_qvel[-2:] * dt
        )  # change to a scalar if the gripper is made symmetric

        if self.has_object:
            # DIY
            if self.hrl_mode:
                # TODO: how to generalize size
                goal_size = 0.025
                achieved_goal_size = 0.025
                obstacle_size = self.object_generator.size_sup

                starting_point = self.cube_starting_point.copy()

                achieved_goal_pos = self.get_xpos(name=self.achieved_name).copy()

                cube_achieved_pos = np.squeeze(achieved_goal_pos.copy())

                cube_obs = np.zeros((length_scale, width_scale, height_scale))

                gripper_link_xpos = self.get_xpos('robot0:gripper_link').copy()
                gripper_link_tuple = (gripper_link_xpos, gripper_link_xpos - gripper_link_size,
                                      gripper_link_xpos + gripper_link_size)

                gripper_finger_xpos_list = [
                    self.get_xpos('robot0:l_gripper_finger_link').copy(),
                    self.get_xpos('robot0:r_gripper_finger_link').copy(),
                ]
                gripper_finger_tuple_list = [
                    (gripper_finger_xpos, gripper_finger_xpos - gripper_finger_size,
                     gripper_finger_xpos + gripper_finger_size) for gripper_finger_xpos in gripper_finger_xpos_list
                ]

                if self.removal_goal is None or self.is_removal_success:
                    goal_xpos = self.global_goal.copy()
                else:
                    goal_xpos = self.removal_goal.copy()
                goal_xpos_tuple = (goal_xpos, goal_xpos - goal_size, goal_xpos + goal_size)

                achieved_goal_xpos = cube_achieved_pos.copy()
                achieved_goal_xpos_tuple = (
                    achieved_goal_xpos, achieved_goal_xpos - achieved_goal_size,
                    achieved_goal_xpos + achieved_goal_size)

                obstacle_xpos_list = [self.get_xpos(obstacle_name).copy() for obstacle_name
                                      in self.obstacle_name_list]
                obstacle_xpos_tuple_list = [
                    (obstacle_xpos, obstacle_xpos - obstacle_size, obstacle_xpos + obstacle_size) for obstacle_xpos in
                    obstacle_xpos_list]
                self._map_object2cube(cube_obs, starting_point,
                                      gripper_link_tuple,
                                      gripper_finger_tuple_list,
                                      goal_xpos_tuple,
                                      achieved_goal_xpos_tuple,
                                      obstacle_xpos_tuple_list,
                                      goal_size,
                                      achieved_goal_size,
                                      obstacle_size,
                                      )

                physical_obs = [grip_pos, gripper_state, grip_velp, gripper_vel]

                achieved_goal_pos = self.get_xpos(self.achieved_name).copy()
                # rotations
                achieved_goal_rot = rotations.mat2euler(self.sim.data.get_site_xmat(self.achieved_name).copy())
                # velocities
                achieved_goal_velp = self.sim.data.get_site_xvelp(self.achieved_name).copy() * dt
                achieved_goal_velr = self.sim.data.get_site_xvelr(self.achieved_name).copy() * dt
                # gripper state
                achieved_goal_rel_pos = achieved_goal_pos - grip_pos
                achieved_goal_velp -= grip_velp

                physical_obs.append(achieved_goal_pos.flatten().copy())
                physical_obs.append(achieved_goal_rot.flatten().copy())
                physical_obs.append(achieved_goal_velp.flatten().copy())
                physical_obs.append(achieved_goal_velr.flatten().copy())
                physical_obs.append(achieved_goal_rel_pos.flatten().copy())
            else:
                object_pos = self.get_xpos("object0").copy()
                # rotations
                object_rot = rotations.mat2euler(self.sim.data.get_site_xmat("object0"))
                # velocities
                object_velp = self.sim.data.get_site_xvelp("object0") * dt
                object_velr = self.sim.data.get_site_xvelr("object0") * dt
                # gripper state
                object_rel_pos = object_pos - grip_pos
                object_velp -= grip_velp
        else:
            object_pos = (
                object_rot
            ) = object_velp = object_velr = object_rel_pos = np.zeros(0)

        # DIY
        achieved_goal = grip_pos.copy()

        # DIY
        if self.hrl_mode:
            obs = np.concatenate(
                [
                    cube_obs.flatten(),
                    np.concatenate(physical_obs),
                ]
            )
        else:
            obs = np.concatenate(
                [
                    grip_pos,
                    np.squeeze(object_pos).ravel(),
                    np.squeeze(object_rel_pos).ravel(),
                    gripper_state,
                    np.squeeze(object_rot).ravel(),
                    np.squeeze(object_velp).ravel(),
                    np.squeeze(object_velr).ravel(),
                    grip_velp,
                    gripper_vel,
                ]
            )

        if self.hrl_mode:
            if self.removal_goal is None or self.is_removal_success:
                goal = self.global_goal.copy()
            else:
                goal = self.removal_goal.copy()
        else:
            goal = self.goal

        return {
            "observation": obs.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": goal.copy(),
        }

    def hrl_reward(self, achieved_goal, goal, info):
        assert self.reward_type == 'dense'

        curr_achi_desi_dist = vector_distance(achieved_goal, goal)
        achi_desi_reward = self.prev_achi_desi_dist - curr_achi_desi_dist
        self.prev_achi_desi_dist = curr_achi_desi_dist

        reward = self.reward_factor * achi_desi_reward

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

    def _state_init(self, goal_xpos: np.ndarray = None):
        achieved_xpos = self.get_xpos("robot0:grip").copy()
        self.prev_achi_desi_dist = vector_distance(achieved_xpos, goal_xpos)

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        global_target_site_id = self.sim.model.site_name2id("global_target")
        removal_target_site_id = self.sim.model.site_name2id("removal_target")
        achieved_site_id = self.sim.model.site_name2id("achieved_site")
        cube_site_id = self.sim.model.site_name2id("cube_site")
        self.sim.model.site_pos[global_target_site_id] = self.global_goal - sites_offset[global_target_site_id]
        self.sim.model.site_pos[removal_target_site_id] = np.array([30, 30, 0.5])
        self.sim.model.site_pos[achieved_site_id] = np.array([30, 30, 0.5])
        self.sim.model.site_pos[cube_site_id] = np.array([30, 30, 0.5])
        self.sim.forward()
