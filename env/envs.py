import numpy as np
from env import fetch_manipulate_env, fetch_manipulate_env_continuous
from env import reach, manipulate
from gym import utils

PREDICATE_THRESHOLD = 0.09  # The minimal threshold to consider two blocks close to each other
DISTANCE_THRESHOLD = 0.05

class FetchManipulateEnv(fetch_manipulate_env.FetchManipulateEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', num_blocks=3, model_path='fetch/stack3.xml'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'object1:joint': [1.25, 0.53, 0.46, 1., 0., 0., 0.],
        }
        fetch_manipulate_env.FetchManipulateEnv.__init__(
            self, model_path, num_blocks=num_blocks, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, predicate_threshold=PREDICATE_THRESHOLD,
            initial_qpos=initial_qpos, reward_type=reward_type, predicates=['close', 'above'],
        )
        utils.EzPickle.__init__(self)

class FetchManipulateEnvContinuous(fetch_manipulate_env_continuous.FetchManipulateEnvContinuous, utils.EzPickle):
    def __init__(self, reward_type='sparse', num_blocks=3, model_path='fetch/stack3.xml'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_manipulate_env_continuous.FetchManipulateEnvContinuous.__init__(
            self, model_path, num_blocks=num_blocks, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, predicate_threshold=PREDICATE_THRESHOLD,
            distance_threshold=DISTANCE_THRESHOLD, initial_qpos=initial_qpos, reward_type=reward_type,
            predicates=['close', 'above'], goals_on_stack_probability=0.6
        )
        utils.EzPickle.__init__(self)

class HandBlockEnv(manipulate.ManipulateEnv, utils.EzPickle):
    def __init__(self, target_position='random', target_rotation='xyz', reward_type='sparse'):
        utils.EzPickle.__init__(self, target_position, target_rotation, reward_type)
        manipulate.ManipulateEnv.__init__(self,
            model_path='hand/manipulate_block.xml', target_position=target_position,
            target_rotation=target_rotation,
            target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)]),
            reward_type=reward_type)
