import os
import copy
import numpy as np
import xml.etree.ElementTree as ET

from gym import error

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: "
        "https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )


def robot_get_obs(sim):
    """Returns all joint positions and velocities associated with
    a robot.
    """
    if sim.data.qpos is not None and sim.model.joint_names:
        names = [n for n in sim.model.joint_names if n.startswith("robot")]
        return (
            np.array([sim.data.get_joint_qpos(name) for name in names]),
            np.array([sim.data.get_joint_qvel(name) for name in names]),
        )
    return np.zeros(0), np.zeros(0)


def ctrl_set_action(sim, action):
    """For torque actuators it copies the action into mujoco ctrl field.
    For position actuators it sets the target relative to the current qpos.
    """
    if sim.model.nmocap > 0:
        _, action = np.split(action, (sim.model.nmocap * 7,))
    if sim.data.ctrl is not None:
        for i in range(action.shape[0]):
            if sim.model.actuator_biastype[i] == 0:
                sim.data.ctrl[i] = action[i]
            else:
                idx = sim.model.jnt_qposadr[sim.model.actuator_trnid[i, 0]]
                sim.data.ctrl[i] = sim.data.qpos[idx] + action[i]


def mocap_set_action(sim, action):
    """The action controls the robot using mocaps. Specifically, bodies
    on the robot (for example the gripper wrist) is controlled with
    mocap bodies. In this case the action is the desired difference
    in position and orientation (quaternion), in world coordinates,
    of the of the target body. The mocap is positioned relative to
    the target body according to the delta, and the MuJoCo equality
    constraint optimizer tries to center the welded body on the mocap.
    """
    if sim.model.nmocap > 0:
        action, _ = np.split(action, (sim.model.nmocap * 7,))
        action = action.reshape(sim.model.nmocap, 7)

        pos_delta = action[:, :3]
        quat_delta = action[:, 3:]

        reset_mocap2body_xpos(sim)
        sim.data.mocap_pos[:] = sim.data.mocap_pos + pos_delta
        sim.data.mocap_quat[:] = sim.data.mocap_quat + quat_delta


def reset_mocap_welds(sim):
    """Resets the mocap welds that we use for actuation."""
    if sim.model.nmocap > 0 and sim.model.eq_data is not None:
        for i in range(sim.model.eq_data.shape[0]):
            if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                sim.model.eq_data[i, :] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    sim.forward()


def reset_mocap2body_xpos(sim):
    """Resets the position and orientation of the mocap bodies to the same
    values as the bodies they're welded to.
    """

    if (
            sim.model.eq_type is None
            or sim.model.eq_obj1id is None
            or sim.model.eq_obj2id is None
    ):
        return
    for eq_type, obj1_id, obj2_id in zip(
            sim.model.eq_type, sim.model.eq_obj1id, sim.model.eq_obj2id
    ):
        if eq_type != mujoco_py.const.EQ_WELD:
            continue

        mocap_id = sim.model.body_mocapid[obj1_id]
        if mocap_id != -1:
            # obj1 is the mocap, obj2 is the welded body
            body_idx = obj2_id
        else:
            # obj2 is the mocap, obj1 is the welded body
            mocap_id = sim.model.body_mocapid[obj2_id]
            body_idx = obj1_id

        assert mocap_id != -1
        sim.data.mocap_pos[mocap_id][:] = sim.data.body_xpos[body_idx]
        sim.data.mocap_quat[mocap_id][:] = sim.data.body_xquat[body_idx]


def get_full_path(path: str):
    if path.startswith("/"):
        fullpath = path
    else:
        fullpath = os.path.join(os.path.dirname(__file__), "assets", path)
    if not os.path.exists(fullpath):
        raise IOError("File {} does not exist".format(fullpath))
    return fullpath


def sample_after_removal(object_name_list: list, object_xpos_list: list, achieved_name: str):
    new_obstacle_name_list = object_name_list.copy()
    assert achieved_name in new_obstacle_name_list
    old_achieved_idx = new_obstacle_name_list.index(achieved_name)

    new_obstacle_name_list.pop(old_achieved_idx)
    object_xpos_list.pop(old_achieved_idx)

    highest_object_idx = None
    highest_object_height = -np.inf
    for new_achieved_idx in range(len(object_xpos_list)):
        new_achieved_xpos = object_xpos_list[new_achieved_idx]
        if new_achieved_xpos[2] > highest_object_height:
            highest_object_idx = new_achieved_idx
            highest_object_height = new_achieved_xpos[2]
    assert highest_object_idx is not None
    new_achieved_name = new_obstacle_name_list[highest_object_idx]
    new_obstacle_name_list.pop(highest_object_idx)
    new_obstacle_name_list.append(achieved_name)

    return new_achieved_name, new_obstacle_name_list


def reset(xml_path='hrl/hrl.xml'):
    fullpath = get_full_path(xml_path)
    tree = ET.parse(fullpath)
    worldbody = tree.find('./worldbody')
    assert worldbody is not None
    original_xpos = [20.0, 20.0, 0.05]
    target = 0
    obstacle = 1
    target_count = 0.0
    obstacle_count = 0.0
    for body in worldbody.findall('body'):
        body_name = body.get('name')
        if body_name.find('object') != -1:
            new_xpos = original_xpos.copy()
            if body_name.find('target') != -1:
                target_count += 1
                new_xpos[target] += target_count
            elif body_name.find('obstacle') != -1:
                obstacle_count += 0.5
                new_xpos[obstacle] += obstacle_count
            body.set('pos', ' '.join([str(i) for i in new_xpos]))
    tree.write(fullpath)


class ObjectGenerator:
    def __init__(self,
                 single_count_sup=7,
                 object_stacked_probability=0.5,
                 random_mode=False,
                 train_upper_mode=False,
                 test_mode=False,
                 stack_mode=False,
                 collect_mode=False,
                 xml_path='hrl/hrl.xml',
                 ):
        self.size_sup = 0.025
        self.height_inf = 0.025
        self.height_sup = self.height_inf + 2 * self.size_sup
        self.xy_dist_sup = 0.16
        self.z_dist_sup = 0.064

        self.object_stacked_probability = object_stacked_probability

        table_xpos = np.array([1.3, 0.75, 0.2])
        table_size = np.array([0.25, 0.35, 0.2])
        self.desktop_lower_boundary = table_xpos - table_size + 5 * self.size_sup
        self.desktop_upper_boundary = table_xpos + table_size - 5 * self.size_sup
        self.desktop_lower_boundary[2] = 0.2 + 0.2 + self.height_inf
        self.desktop_upper_boundary[2] = 0.2 + 0.2 + self.height_sup

        self.initial_xpos_origin = np.array([20.0, 20.0, 0.0])
        self.initial_xpos_size = np.array([5.0, 5.0, 0])

        self.qpos_postfix = np.array([1.0, 0.0, 0.0, 0.0])

        self.single_count_sup = single_count_sup + 1

        self.random_mode = random_mode
        self.train_upper_mode = train_upper_mode
        self.test_mode = test_mode
        self.stack_mode = stack_mode
        self.collect_mode = collect_mode

        self.global_achieved_name = 'target_object'

        self.object_name_list = []
        self.obstacle_name_list = []
        self.object_count_sup = None
        self.init_total_obstacle(xml_path=xml_path)
        if not self.test_mode:
            object_remove_name_list = []
            for name in self.object_name_list:
                if 'target' in name and name != 'target_object':
                    object_remove_name_list.append(name)

            for name in object_remove_name_list:
                self.object_name_list.remove(name)

        self.step = 0.05
        self.delta_obstacle_qpos_list = [
            np.r_[np.array([1.44, 0.88, 0.425]) - np.array([1.34, 0.88, 0.425]), self.qpos_postfix],
            np.r_[np.array([1.24, 0.88, 0.425]) - np.array([1.34, 0.88, 0.425]), self.qpos_postfix],

            np.r_[np.array([1.30 - 0.20, 0.75 + 0.20, 0.425]) - np.array([1.30 - 0.20, 0.75 - 0.20, 0.425]), self.qpos_postfix],
            # np.r_[np.array([1.30 + 0.20, 0.75 - 0.20, 0.425]) - np.array([1.30 - 0.20, 0.75 - 0.20, 0.425]), self.qpos_postfix],
            np.r_[np.array([1.30 + 0.20, 0.75 + 0.20, 0.425]) - np.array([1.30 - 0.20, 0.75 - 0.20, 0.425]), self.qpos_postfix],

            # np.r_[np.array([1.30, 0.75 - 0.20, 0.425]) - np.array(
            #     [1.30 - 0.20, 0.75 - 0.20, 0.425]), self.qpos_postfix],
            # np.r_[np.array([1.30, 0.75 - 0.20, 0.475]) - np.array(
            #     [1.30 - 0.20, 0.75 - 0.20, 0.425]), self.qpos_postfix],
            np.r_[np.array([1.30, 0.75 + 0.20, 0.425]) - np.array(
                [1.30 - 0.20, 0.75 - 0.20, 0.425]), self.qpos_postfix],
            np.r_[np.array([1.30, 0.75 + 0.20, 0.475]) - np.array(
                [1.30 - 0.20, 0.75 - 0.20, 0.425]), self.qpos_postfix],

            np.r_[[-self.step * 1, 0, 0], self.qpos_postfix],
            np.r_[[self.step * 1, 0, 0], self.qpos_postfix],
            np.r_[[0, 0, 2 * self.step], self.qpos_postfix],
            np.r_[[0, 0, 3 * self.step], self.qpos_postfix],
            np.r_[[-self.step, 0, 1 * self.step], self.qpos_postfix],
            np.r_[[self.step, 0,  1 * self.step], self.qpos_postfix],
            np.r_[[0, -self.step, 0], self.qpos_postfix],
            np.r_[[0, self.step, 0], self.qpos_postfix],
        ]

        # filter the first two elements
        if self.collect_mode:
            self.delta_obstacle_qpos_list = self.delta_obstacle_qpos_list[2:]

        self.obstacle_count = 3
        if self.stack_mode:
            self.obstacle_count = 2
        elif self.collect_mode:
            self.obstacle_count = 3 + 1

        self.max_stack_count = 4
        self.possible_stack_qpos_list = []
        self.stack_qpos_postfix = np.array([0.0, 0.0, 0.0, 0.0])
        for stack_count in range(2, self.max_stack_count + 1):
            stack_qpos_list = []
            for i in range(1, stack_count):
                stack_qpos_list.append(np.r_[0, 0, -i * self.step, self.stack_qpos_postfix])
            self.possible_stack_qpos_list.append(stack_qpos_list)
        assert len(self.possible_stack_qpos_list) == self.max_stack_count - 1

        self.test_scenario_start_idx = 0
        self.test_scenario_end_idx   = 5 + 1

        self.test_scenario_name_list = None
        self.achieved_name_list = None
        self.test_scenario_xpos_list = None
        self.test_scenario_goal_list = None
        self.test_scenario_best_motion_list = None
        self.test_count = 0
        self.achieved_name_indicate = 0
        self.reset_scenario_list(self.test_scenario_start_idx, self.test_scenario_end_idx)

    def init_total_obstacle(self, xml_path='hrl/hrl.xml'):
        self.object_count_sup = 0
        fullpath = get_full_path(xml_path)
        tree = ET.parse(fullpath)
        worldbody = tree.find('./worldbody')
        assert worldbody is not None
        for body in worldbody.findall('body'):
            body_name = body.get('name')
            if body_name.find('object') != -1:
                object_name = body_name
                self.object_name_list.append(object_name)
                self.object_count_sup += 1
            if body_name.find('obstacle') != -1:
                obstacle_name = body_name
                self.obstacle_name_list.append(obstacle_name)

    def sample_one_qpos_on_table(self, achieved_xpos: np.ndarray):
        object_qpos = np.r_[achieved_xpos, self.qpos_postfix]
        object_xpos = object_qpos[:3]

        object_xpos[2] = self.desktop_lower_boundary[2]

        # 1.5 * size_sup: ensure obstacle will not cover achieved_object
        if np.random.uniform() < 0.5:  # negative offset
            delta_xy_dist = np.random.uniform(-self.xy_dist_sup, -2.5 * self.size_sup, 2)
        else:  # positive offset
            delta_xy_dist = np.random.uniform(2.5 * self.size_sup, self.xy_dist_sup, 2)
        delta_z_dist = np.random.uniform(0, self.z_dist_sup)
        object_xpos[: 2] += delta_xy_dist
        object_xpos[: 2] = np.where(object_xpos[: 2] >= self.desktop_lower_boundary[: 2],
                                    object_xpos[: 2],
                                    self.desktop_lower_boundary[: 2])
        object_xpos[: 2] = np.where(object_xpos[: 2] <= self.desktop_upper_boundary[: 2],
                                    object_xpos[: 2],
                                    self.desktop_upper_boundary[: 2])
        object_xpos[2] += delta_z_dist
        return object_qpos

    def stack_setup(self):
        stack_qpos_list = self.possible_stack_qpos_list[np.random.randint(len(self.possible_stack_qpos_list))].copy()
        return stack_qpos_list

    def sample_objects(self):
        if self.test_mode:
            return self.test_set_objects()

        achieved_name = 'target_object'

        tmp_object_name_list = self.object_name_list.copy()
        tmp_object_name_list.remove(achieved_name)
        object_name_list = []
        object_qpos_list = []
        obstacle_name_list = []
        obstacle_xpos_list = []

        if self.stack_mode or self.collect_mode:
            if self.random_mode:
                achieved_qpos = np.r_[np.random.uniform(self.desktop_lower_boundary, self.desktop_upper_boundary),
                                      self.qpos_postfix]
                if self.collect_mode:
                    obstacle_count = np.random.randint(1, 4 + 1)
                elif self.stack_mode:
                    # obstacle_count = np.random.randint(1, 2 + int(not self.train_upper_mode) + 1)
                    obstacle_count = 2
                else:
                    raise NotImplementedError
            else:
                achieved_qpos = np.r_[np.array([1.3, 0.75, 0.425]), self.qpos_postfix]
                obstacle_count = self.obstacle_count
        elif self.random_mode:
            if np.random.uniform() < self.object_stacked_probability:
                achieved_qpos = np.r_[
                    np.random.uniform(self.desktop_lower_boundary[:2], self.desktop_upper_boundary[:2]),
                    self.desktop_lower_boundary[2],
                    self.qpos_postfix,
                ]
                stack_qpos_list = self.stack_setup()
                achieved_qpos[2] += self.step * len(stack_qpos_list)
                for i in range(len(stack_qpos_list)):
                    object_name_list.append(tmp_object_name_list[0])
                    obstacle_name_list.append(tmp_object_name_list[0])
                    tmp_object_name_list.pop(0)

                    obstacle_qpos = achieved_qpos + stack_qpos_list[i]
                    object_qpos_list.append(obstacle_qpos.copy())
                    obstacle_xpos_list.append(obstacle_qpos[:3].copy())

                if self.train_upper_mode:
                    swap_index = np.random.randint(len(obstacle_xpos_list))
                    tmp_achieved_xpos = obstacle_xpos_list[swap_index].copy()

                    obstacle_xpos_list.pop(swap_index)
                    obstacle_xpos_list.insert(swap_index, achieved_qpos[:3].copy())
                    object_qpos_list.pop(swap_index)
                    object_qpos_list.insert(swap_index, achieved_qpos.copy())

                    achieved_qpos = np.r_[
                        tmp_achieved_xpos.copy(),
                        self.qpos_postfix,
                    ]

                obstacle_count = np.random.randint(self.single_count_sup
                                                   - 1  # recover + 1 in __init__
                                                   - len(stack_qpos_list)
                                                   - 1  # old achieve
                                                   )
            else:
                obstacle_count = np.random.randint(1, self.single_count_sup)
                achieved_qpos = np.r_[
                    np.random.uniform(self.desktop_lower_boundary, self.desktop_upper_boundary), self.qpos_postfix]
        else:
            obstacle_count = self.obstacle_count
            self.obstacle_count = 3 + (self.obstacle_count + 1) % 3
            achieved_qpos = np.r_[np.array([1.34, 0.88, 0.425]), self.qpos_postfix]

        object_name_list.insert(0, achieved_name)
        object_qpos_list.insert(0, achieved_qpos.copy())

        if self.random_mode or self.collect_mode:
            new_obstacle_name_list = list(np.random.choice(tmp_object_name_list, size=obstacle_count, replace=False))
        else:
            new_obstacle_name_list = tmp_object_name_list[:obstacle_count].copy()

        object_name_list.extend(new_obstacle_name_list.copy())
        obstacle_name_list.extend(new_obstacle_name_list.copy())

        # DIY
        delta_obstacle_qpos_list = self.delta_obstacle_qpos_list[: obstacle_count].copy()

        if not self.random_mode:
            for delta_obstacle_qpos in delta_obstacle_qpos_list:
                object_qpos_list.append(achieved_qpos.copy() + delta_obstacle_qpos)
                obstacle_xpos_list.append((achieved_qpos.copy() + delta_obstacle_qpos)[:3])

            return achieved_name, dict(zip(object_name_list, object_qpos_list)), dict(
                zip(obstacle_name_list, obstacle_xpos_list))

        for _ in np.arange(obstacle_count):
            obstacle_qpos = self.sample_one_qpos_on_table(achieved_xpos=achieved_qpos[:3])
            object_qpos_list.append(obstacle_qpos)
            obstacle_xpos_list.append(obstacle_qpos[:3])

        assert len(object_name_list) == len(object_qpos_list) and len(obstacle_name_list) == len(obstacle_xpos_list)

        return achieved_name, dict(zip(object_name_list, object_qpos_list)), dict(
            zip(obstacle_name_list, obstacle_xpos_list))

    def resample_obstacles(self, object_name_list: list, obstacle_count: int):
        if self.test_mode:
            return self.test_resample_obstacles(object_name_list=object_name_list, obstacle_count=obstacle_count)

        if self.random_mode:
            achieved_xpos = np.r_[
                np.random.uniform(self.desktop_lower_boundary[:2], self.desktop_upper_boundary[:2]),
                self.desktop_lower_boundary[2],
            ]
        else:
            achieved_xpos = np.array([1.34, 0.88, 0.425])

        achieved_qpos = np.r_[achieved_xpos, self.qpos_postfix]
        object_qpos_list = [achieved_qpos.copy()]

        delta_obstacle_qpos_list = self.delta_obstacle_qpos_list[: obstacle_count].copy()

        if not self.random_mode:
            for delta_obstacle_qpos in delta_obstacle_qpos_list:
                object_qpos_list.append(achieved_qpos.copy() + delta_obstacle_qpos)
            return dict(zip(object_name_list, object_qpos_list))

        for _ in np.arange(obstacle_count):
            obstacle_qpos = self.sample_one_qpos_on_table(achieved_xpos=achieved_xpos.copy())
            object_qpos_list.append(obstacle_qpos)

        return dict(zip(object_name_list, object_qpos_list))

    def set_global_achieved_name(self, achieved_name: str):
        self.global_achieved_name = achieved_name

    def test_set_objects(self):
        assert self.test_mode
        scenario_name = self.test_scenario_name_list[self.test_count]
        # print(f'Description of scenario: {scenario_name}')
        scenario_xpos_dict = self.test_scenario_xpos_list[self.test_count]
        achieved_name_list = self.achieved_name_list[self.test_count]
        assert self.achieved_name_indicate < len(achieved_name_list)
        achieved_name = achieved_name_list[self.achieved_name_indicate]

        achieved_xpos = scenario_xpos_dict['target_object'].copy()
        if self.random_mode:  # used to make algorithm stochastic
            achieved_xpos[:2] = np.random.uniform(self.desktop_lower_boundary[:2], self.desktop_upper_boundary[:2])

        self.global_achieved_name = copy.deepcopy(achieved_name)
        self.achieved_name_indicate += 1
        if self.achieved_name_indicate >= len(achieved_name_list):
            self.achieved_name_indicate = 0
            self.test_count = (self.test_count + 1) % len(self.test_scenario_xpos_list)

        tmp_object_name_list = self.object_name_list.copy()
        target_name_list = []
        for name in tmp_object_name_list:
            if 'target' in name:
                target_name_list.append(name)
        for name in target_name_list:
            tmp_object_name_list.remove(name)

        object_name_list = [achieved_name]
        object_qpos_list = [np.r_[achieved_xpos.copy(), self.qpos_postfix]]
        obstacle_name_list = []
        obstacle_xpos_list = []

        assert len(scenario_xpos_dict['obstacle_object']) <= len(tmp_object_name_list)
        for delta_obstacle_xpos in scenario_xpos_dict['obstacle_object']:
            obstacle_xpos = achieved_xpos + delta_obstacle_xpos
            object_qpos_list.append(np.r_[obstacle_xpos.copy(), self.qpos_postfix])
            obstacle_xpos_list.append(obstacle_xpos.copy())

        object_name_list.extend(tmp_object_name_list[:len(scenario_xpos_dict['obstacle_object'])])
        obstacle_name_list.extend(tmp_object_name_list[:len(scenario_xpos_dict['obstacle_object'])])

        return achieved_name, dict(zip(object_name_list, object_qpos_list)), dict(
            zip(obstacle_name_list, obstacle_xpos_list))

    def test_resample_obstacles(self, object_name_list: list, obstacle_count: int):
        scenario_xpos_dict = self.test_scenario_xpos_list[self.test_count - 1]
        assert len(scenario_xpos_dict['obstacle_object']) == obstacle_count

        achieved_xpos = scenario_xpos_dict['target_object'].copy()
        if self.random_mode:
            achieved_xpos[:2] = np.random.uniform(self.desktop_lower_boundary[:2], self.desktop_upper_boundary[:2])

        object_qpos_list = [np.r_[achieved_xpos.copy(), self.qpos_postfix]]

        for delta_obstacle_xpos in scenario_xpos_dict['obstacle_object']:
            obstacle_xpos = achieved_xpos + delta_obstacle_xpos
            object_qpos_list.append(np.r_[obstacle_xpos.copy(), self.qpos_postfix])

        return dict(zip(object_name_list, object_qpos_list))

    def test_set_goal(self) -> np.ndarray:
        assert self.test_mode
        goal_xpos = self.test_scenario_goal_list[self.test_count - 1]

        return goal_xpos.copy()

    def set_target_shape(self, shape) -> None:
        if shape == 'box':
            target_name = 'target_object'
        elif shape == 'sphere':
            target_name = 'target_object_sphere'
        elif shape == 'ellipsoid':
            target_name = 'target_object_ellipsoid'
        elif shape == 'capsule':
            target_name = 'target_object_capsule'
        else:
            raise NotImplementedError

        self.global_achieved_name = target_name
        for idx in range(len(self.achieved_name_list)):
            self.achieved_name_list[idx] = [target_name]

    def reset_scenario_list(self, start_idx, end_idx):
        self.test_scenario_start_idx = start_idx
        self.test_scenario_end_idx = end_idx
        self.test_scenario_name_list = [
            # easy
            'some above target object',
            'one obstacle above target object and one stacked obstacles',
            # medium
            'four direction block',
            'all stack above target object',
            # hard
            'cover target object densely',
            'demo',
        ][self.test_scenario_start_idx: self.test_scenario_end_idx]
        self.achieved_name_list = [
            ['target_object'],
            ['target_object'],

            ['target_object'],
            ['target_object'],

            ['target_object'],
            ['target_object'],
        ][self.test_scenario_start_idx: self.test_scenario_end_idx]
        self.test_scenario_xpos_list = [
            {
                'target_object':
                    np.array([1.30, 0.88, 0.425 + self.step * 2]),
                'obstacle_object': [
                    np.array([1.30, 0.88, 0.425 + self.step * 0]) - np.array([1.30, 0.88, 0.425 + self.step * 2]),
                    np.array([1.30, 0.88, 0.425 + self.step * 1]) - np.array([1.30, 0.88, 0.425 + self.step * 2]),
                    np.array([1.30, 0.88, 0.425 + self.step * 3]) - np.array([1.30, 0.88, 0.425 + self.step * 2]),

                    np.array([1.30 - self.step * 1, 0.88, 0.425]) - np.array([1.30, 0.88, 0.425 + self.step * 2]),
                    np.array([1.30 + self.step * 1, 0.88, 0.425]) - np.array([1.30, 0.88, 0.425 + self.step * 2]),
                    np.array([1.30, 0.88 - self.step * 1, 0.425]) - np.array([1.30, 0.88, 0.425 + self.step * 2]),
                    np.array([1.30, 0.88 + self.step * 1, 0.425]) - np.array([1.30, 0.88, 0.425 + self.step * 2]),
                ],
            },
            {
                'target_object':
                    np.array([1.30, 0.63, 0.425 + self.step * 0]),
                'obstacle_object': [
                    np.array([1.30, 0.63, 0.425 + self.step * 1]) - np.array([1.30, 0.63, 0.425 + self.step * 0]),

                    np.array([1.30, 0.83, 0.425 + self.step * 0]) - np.array([1.30, 0.63, 0.425 + self.step * 0]),
                    np.array([1.30, 0.83, 0.425 + self.step * 1]) - np.array([1.30, 0.63, 0.425 + self.step * 0]),
                ],
            },
            {
                'target_object':
                    np.array([1.30, 0.80, 0.425]),
                'obstacle_object': [
                    np.array([1.30, 0.80, 0.425 + self.step * 1]) - np.array([1.30, 0.80, 0.425]),
                    np.array([1.30, 0.80, 0.425 + self.step * 2]) - np.array([1.30, 0.80, 0.425]),

                    np.array([1.18 + self.step * 0, 0.50, 0.425 + self.step * 1]) - np.array([1.30, 0.80, 0.425]),
                    np.array([1.18 + self.step * 0, 0.50, 0.425 + self.step * 0]) - np.array([1.30, 0.80, 0.425]),
                    np.array([1.18 - self.step * 1, 0.50, 0.425 + self.step * 0]) - np.array([1.30, 0.80, 0.425]),
                    np.array([1.18 + self.step * 1, 0.50, 0.425 + self.step * 0]) - np.array([1.30, 0.80, 0.425]),
                ],
            },
            {
                'target_object':
                    np.array([1.30, 0.65, 0.425]),
                'obstacle_object': [
                    np.array([1.30 - self.step * 3, 0.65, 0.425 + self.step * 0]) - np.array([1.30, 0.65, 0.425]),
                    np.array([1.30 + self.step * 3, 0.65, 0.425 + self.step * 0]) - np.array([1.30, 0.65, 0.425]),
                    np.array([1.30, 0.65 - self.step * 3, 0.425 + self.step * 0]) - np.array([1.30, 0.65, 0.425]),
                    np.array([1.30, 0.65 + self.step * 3, 0.425 + self.step * 0]) - np.array([1.30, 0.65, 0.425]),

                    np.array([1.30, 0.65, 0.425 + self.step]) - np.array([1.30, 0.65, 0.425]),
                ],
            },
            {
                'target_object':
                    np.array([1.30, 0.50, 0.425]),
                'obstacle_object': [
                    np.array([1.30 - self.step * 1, 0.50,                 0.425]) - np.array([1.30, 0.50, 0.425]),
                    np.array([1.30 + self.step * 1, 0.50,                 0.425]) - np.array([1.30, 0.50, 0.425]),

                    np.array([1.30,                 0.50,                 0.425 + self.step * 1]) - np.array([1.30, 0.50, 0.425]),
                    np.array([1.30,                 0.50,                 0.425 + self.step * 2]) - np.array([1.30, 0.50, 0.425]),
                ],
            },
            {
                'target_object':
                    np.array([1.28, 0.82, 0.425]),
                'obstacle_object': [
                    np.array([1.28, 0.82, 0.425 + self.step * 1]) - np.array([1.28, 0.82, 0.425]),
                    np.array([1.28, 0.82, 0.425 + self.step * 2]) - np.array([1.28, 0.82, 0.425]),

                    np.array([1.28 - 0.15, 0.82 - 0.00, 0.425]) - np.array([1.28, 0.82, 0.425]),
                    np.array([1.28 - 0.00, 0.82 - 0.15, 0.425]) - np.array([1.28, 0.82, 0.425]),
                    np.array([1.28 + 0.00, 0.82 + 0.15, 0.425]) - np.array([1.28, 0.82, 0.425]),
                    np.array([1.28 + 0.15, 0.82 + 0.00, 0.425]) - np.array([1.28, 0.82, 0.425]),
                ],
            },
        ][self.test_scenario_start_idx: self.test_scenario_end_idx]
        self.test_scenario_goal_list = [
            np.array([1.30, 0.75, 0.540]),
            np.array([1.30, 0.75, 0.540]),

            np.array([1.30, 0.75, 0.540]),
            np.array([1.30, 0.75, 0.540]),

            np.array([1.30, 0.75, 0.5400]),
            np.array([1.30, 0.75, 0.540]),
        ][self.test_scenario_start_idx: self.test_scenario_end_idx]
        self.test_scenario_best_motion_list = [
            1.5,
            3,
            3,
        ]

        assert len(self.test_scenario_name_list) == len(self.achieved_name_list),\
            f'Left length: {len(self.test_scenario_name_list)} != Right length: {len(self.achieved_name_list)}'
        assert len(self.test_scenario_name_list) == len(self.test_scenario_xpos_list),\
            f'Left length: {len(self.test_scenario_name_list)} != Right length: {len(self.test_scenario_xpos_list)}'
        assert len(self.test_scenario_name_list) == len(self.test_scenario_goal_list),\
            f'Left length: {len(self.test_scenario_name_list)} != Right length: {len(self.test_scenario_goal_list)}'

        self.test_count = 0
        self.achieved_name_indicate = 0
