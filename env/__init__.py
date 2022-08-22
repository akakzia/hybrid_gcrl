from gym.envs.registration import register

import sys
from functools import reduce


def str_to_class(str):
    return reduce(getattr, str.split("."), sys.modules[__name__])


# Robotics
# ----------------------------------------
num_blocks = 3

register(id='FetchManipulate3Objects-v0',
         entry_point='env.envs:FetchManipulateEnv',
         kwargs={'reward_type': 'sparse'},
         max_episode_steps=100,)

register(id='FetchManipulate4Objects-v0',
         entry_point='env.envs:FetchManipulateEnv',
         kwargs={'reward_type': 'sparse',
                 'num_blocks': 4,
                 'model_path': 'fetch/stack4.xml'
                 },
         max_episode_steps=100,)

register(id='FetchManipulate5Objects-v0',
         entry_point='env.envs:FetchManipulateEnv',
         kwargs={'reward_type': 'sparse',
                 'num_blocks': 5,
                 'model_path': 'fetch/stack5.xml'
                 },
         max_episode_steps=200,)

register(id='FetchManipulate5ObjectsContinuous-v0',
         entry_point='env.envs:FetchManipulateEnvContinuous',
         kwargs={'reward_type': 'incremental',
                 'num_blocks': 5,
                 'model_path': 'fetch/stack5_with_targets.xml'
                 },
         max_episode_steps=200,)

register(id='FetchManipulate3ObjectsContinuous-v0',
         entry_point='env.envs:FetchManipulateEnvContinuous',
         kwargs={'reward_type': 'incremental',
                 'num_blocks': 3,
                 'model_path': 'fetch/stack3_with_targets.xml'
                 },
         max_episode_steps=100,)

register(id='FetchManipulate1ObjectContinuous-v0',
         entry_point='env.envs:FetchManipulateEnvContinuous',
         kwargs={'reward_type': 'incremental',
                 'num_blocks': 1,
                 'model_path': 'fetch/stack1_with_targets.xml'
                 },
         max_episode_steps=50,) 

# Hand
register(
id='HandReach-v2',
entry_point='env.reach:HandReachEnv',
kwargs={
        'reward_type': 'incremental'
        },
max_episode_steps=50,
)

register(
id='HandManipulateBlockRotateZ-v1',
entry_point='env.envs:HandBlockEnv',
kwargs={
        'reward_type': 'sparse',
        'target_position': 'ignore',
        'target_rotation': 'z'
        },
max_episode_steps=100,
)