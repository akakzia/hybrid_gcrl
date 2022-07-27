import numpy as np
from utils import get_idxs_per_relation
from mpi4py import MPI


N_POINTS = 42
QUEUE_LENGTH = 1000
EPSILON = 0.1 # When sampling from buffer, proba to sample randomly (not using LP)


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.n_classes = 3
        self.goal_dim = args.env_params['goal']

        self.init_stats()

    def sample_goal(self, n_goals=2, evaluation=False):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation:
            return np.array([0, 1, 2])
        else:
            # decide whether to self evaluate
            self_eval = False
            goals = np.random.choice(range(self.n_classes), size=n_goals)

        return goals, self_eval
    
    def update(self, episodes):
        """ Update the successes and failures """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = []
            for eps in all_episodes:
                all_episode_list += eps
            
            for e in all_episode_list:
                if e['self_eval']:
                    if (e['rewards'][-1] == 5.):
                        success = 1
                    else:
                        success = 0
                    self.successes_and_failures[e['goal_class']].append(success)
                    # Make sure not to excede queue length
                    if len(self.successes_and_failures[e['goal_class']]) > QUEUE_LENGTH:
                        self.successes_and_failures = self.successes_and_failures[-QUEUE_LENGTH:]

        return episodes

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        for i in np.arange(1, self.n_classes+1):
            self.stats['Eval_SR_{}'.format(i)] = []
            self.stats['Av_Rew_{}'.format(i)] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        for g_id in np.arange(1, len(av_res) + 1):
            self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id-1])
            self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id-1])
