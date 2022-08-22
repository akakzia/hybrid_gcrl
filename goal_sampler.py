import numpy as np
from utils import get_idxs_per_relation
from mpi4py import MPI
from mpi_utils.entropy_estimators import get_h


N_POINTS = 42
QUEUE_LENGTH = 1000
EPSILON = 0.1 # When sampling from buffer, proba to sample randomly (not using LP)


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.goal_dim = args.env_params['goal']

        self.discovered_goals = []
        self.len_goal_buffer = args.len_goal_buffer

        self.external_goal_generation_ratio = args.external_goal_generation_ratio

        self.init_stats()

    def sample_goal(self, n_goals=2, bootstrapping=False, evaluation=False):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation:
            # Evaluate one goal for discovered goal 
            # One goal for external goal
            try:
                eval_intern_goal = np.array(self.discovered_goals)[np.random.choice(np.arange(len(self.discovered_goals)))]
                return [eval_intern_goal, None]
            except ValueError:
                return [None, None]
        else:
            # Decide whether to use extrinsic goals or intrinsic goals
            external = np.random.uniform() < self.external_goal_generation_ratio
            # Goals will be generated externally in the env interface
            # Agent will perform a bootstrapping phase where it performs arbitrary action independent of goals
            # So goals here are not important
            if external or bootstrapping: 
                goals = np.array([None for _ in range(n_goals)])
                
                return goals, external 

            # Goals are generated internally using the list of discovere goals
            else:
                ids = np.random.choice(np.arange(len(self.discovered_goals)), size=n_goals)
                goals = np.array(self.discovered_goals)[ids]

            return goals, external
    
    def update(self, episodes):
        """ Update the successes and failures """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            # self.discovered_goals += [ag for eps in all_episodes for e in eps for ag in np.unique(np.around(e['ag'][-10:], decimals=3), axis=0)]
            # Only the last achieved goal is considered as a new discovered goal.
            self.discovered_goals += [e['ag'][-1] for eps in all_episodes for e in eps] 
            
        self.sync()
        return episodes

    def sync(self):
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals[-self.len_goal_buffer:], root=0)

    def init_stats(self):
        self.stats = dict()
        self.stats['sr_internal'] = []
        self.stats['sr_external'] = []
        self.stats['r_internal'] = []
        self.stats['r_external'] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['entropy'] = []
        # self.stats['av_rew'] = []
        self.stats['global_sr'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_sr, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        # self.stats['av_rew'].append(av_rew)
        self.stats['sr_internal'].append(av_sr[0])
        self.stats['sr_external'].append(av_sr[1])
        self.stats['r_internal'].append(av_sr[0])
        self.stats['r_external'].append(av_sr[1])
        # Compute entropy of discovered goal distibution in the running epoch
        running_data = self.discovered_goals[-1900:]
        # Normalize
        running_data = (running_data - np.mean(running_data, axis=0)) / np.std(running_data, axis=0)
        h = get_h(np.array(running_data), k=5)
        self.stats['entropy'].append(h) 