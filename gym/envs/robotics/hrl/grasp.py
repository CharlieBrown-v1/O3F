import os
from gym import utils
from gym.envs.robotics import fetch_env


MODEL_XML_PATH = os.path.join("hrl", "hrl.xml")


class GraspEnv(fetch_env.FetchEnv, utils.EzPickle):
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
            single_count_sup=7,
            target_in_air_probability=0.5,
            object_stacked_probability=0.5,
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
        goal[2] = self.height_offset

        assert self.target_in_the_air
        if self.demo_mode:
            goal[2] += self.np_random.uniform(0.1, 0.2)
        elif self.np_random.uniform() < self.target_in_air_probability:
            goal[2] += self.np_random.uniform(self.distance_threshold, 0.3)

        self.reset_removal(goal=goal.copy(), is_removal=is_removal)

        return goal.copy()
