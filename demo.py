import torch
from rl_modules.rl_agent import RLAgent
import env
import gym
import numpy as np
from rollout import RolloutWorker
import json
from types import SimpleNamespace
from goal_sampler import GoalSampler
import  random
from mpi4py import MPI
from arguments import get_args

def get_env_params(env):
    obs = env.reset()
    # close the environment
    params = {'obs': obs['observation'].shape[0], 'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0], 'action_max': env.action_space.high[0],
              'max_timesteps': env._max_episode_steps}
    return params

if __name__ == '__main__':
    num_eval = 1
    path = '/home/ahmed/'
    model_path = path + 'model_110.pt'

    args = get_args()

    args.env_name = 'HandManipulateEggFull-v2'

    if args.env_name == 'FetchManipulate3ObjectsContinuous-v0':
        args.architecture = 'interaction_network'
    else:
        args.architecture = 'flat'

    # Make the environment
    env = gym.make(args.env_name)

    # set random seeds for reproduce
    args.seed = np.random.randint(1e6)
    env.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())

    args.env_params = get_env_params(env)

    goal_sampler = GoalSampler(args)

    # create the sac agent to interact with the environment
    policy = RLAgent(args, env.compute_reward, goal_sampler)
    policy.load(model_path, args)

    # def rollout worker
    rollout_worker = RolloutWorker(env, policy, goal_sampler,  args)

    eval_goals = goal_sampler.sample_goal(n_goals=10, evaluation=True)

    all_results = []
    for i in range(num_eval):
        episodes = rollout_worker.generate_rollout(eval_goals,  external=True, true_eval=True, animated=True)
        results = np.array([e['rewards'][-1] == 1. for e in episodes])
        all_results.append(results)

    results = np.array(all_results)
    print('Av Success Rate: {}'.format(results.mean()))