import torch
import numpy as np
from mpi_utils.mpi_utils import sync_networks
from rl_modules.replay_buffer import ReplayBuffer
from rl_modules.networks import QNetworkFlat, GaussianPolicyFlat
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler
from updates import update_ddpg


"""
SAC with HER (MPI-version)
"""

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class RLAgent:
    def __init__(self, args, compute_rew, goal_sampler):

        self.args = args
        self.env_params = args.env_params

        self.goal_sampler = goal_sampler

        self.total_iter = 0

        self.freq_target_update = args.freq_target_update

        # create the network
        self.architecture = self.args.architecture

        if self.architecture == 'flat':
            from rl_modules.flat_models import FlatSemantic
            self.model = FlatSemantic(self.env_params)
        elif self.architecture == 'interaction_network':
            from rl_modules.interaction_models import InSemantic
            self.model = InSemantic(self.env_params, args)
        elif self.architecture == 'interaction_network_2':
            from rl_modules.interaction_models_v2 import GnSemantic
            self.model = GnSemantic(self.env_params, args)
        elif self.architecture == 'full_gn':
            from rl_modules.gn_models import GnSemantic
            self.model = GnSemantic(self.env_params, args)
        elif self.architecture == 'relation_network':
            from rl_modules.rn_models import RnSemantic
            self.model = RnSemantic(self.env_params, args)
        elif self.architecture == 'deep_sets':
            from rl_modules.deepsets_models import DsSemantic
            self.model = DsSemantic(self.env_params, args)
        else:
            raise NotImplementedError

        # if use GPU
        if self.args.cuda:
            self.model.actor.cuda()
            self.model.critic.cuda()
            self.model.critic_target.cuda()
        # sync the networks across the CPUs
        sync_networks(self.model.critic)
        sync_networks(self.model.actor)
        hard_update(self.model.critic_target, self.model.critic)
        sync_networks(self.model.critic_target)

        # create the optimizer
        self.policy_optim = torch.optim.Adam(list(self.model.actor.parameters()), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(list(self.model.critic.parameters()), lr=self.args.lr_critic)
        
        # create the normalizer
        self.o_norm = normalizer(size=self.env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=self.env_params['goal'], default_clip_range=self.args.clip_range)

        # her sampler
        self.her_module = her_sampler(self.args, compute_rew)

        # create the replay buffer
        self.buffer = ReplayBuffer(env_params=self.env_params,
                                  buffer_size=self.args.buffer_size,
                                  sample_func=self.her_module.sample_her_transitions,
                                  goal_sampler=self.goal_sampler
                                  )

    def act(self, obs, ag, g, no_noise):
        with torch.no_grad():
            # normalize policy inputs
            obs_norm = self.o_norm.normalize(obs)
            ag_norm = torch.tensor(self.g_norm.normalize(ag), dtype=torch.float32).unsqueeze(0)
            g_norm = torch.tensor(self.g_norm.normalize(g), dtype=torch.float32).unsqueeze(0)

            obs_tensor = torch.tensor(obs_norm, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                obs_tensor = obs_tensor.cuda()
                g_norm = g_norm.cuda()
                ag_norm = ag_norm.cuda()
            self.model.policy_forward_pass(obs_tensor, ag_norm, g_norm, no_noise=True)
            if self.args.cuda:
                action = self.model.pi_tensor.cpu().numpy()[0]
            else:
                action = self.model.pi_tensor.numpy()[0]
            
            if not no_noise:
                # add the gaussian
                action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
                action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
                # random actions...
                random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                                    size=self.env_params['action'])
                # choose if use the random actions
                action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)

                
        return action.copy()
    
    def store(self, episodes):
        self.buffer.store_episode(episode_batch=episodes)

    def train(self):
        # train the network
        self.total_iter += 1
        self._update_network()

        # soft update
        if self.total_iter % self.freq_target_update == 0:
            self._soft_update_target_network(self.model.critic_target, self.model.critic)
                

    # update the normalizer
    def _update_normalizer(self, episode):
        mb_obs = episode['obs']
        mb_ag = episode['ag']
        mb_g = episode['g']
        mb_actions = episode['act']
        mb_obs_next = mb_obs[1:, :]
        mb_ag_next = mb_ag[1:, :]
        # get the number of normalization transitions
        num_transitions = mb_actions.shape[0]
        # create the new buffer to store them
        buffer_temp = {'obs': np.expand_dims(mb_obs, 0),
                       'ag': np.expand_dims(mb_ag, 0),
                       'g': np.expand_dims(mb_g, 0),
                       'actions': np.expand_dims(mb_actions, 0),
                       'obs_next': np.expand_dims(mb_obs_next, 0),
                       'ag_next': np.expand_dims(mb_ag_next, 0),
                       }

        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions)
        obs, g = transitions['obs'], transitions['g']
        # pre process the obs and g
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        # update
        self.o_norm.update(transitions['obs'])
        # recompute the stats
        self.o_norm.recompute_stats()

        self.g_norm.update(transitions['g'])
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample from buffer, this is done with LP is multi-head is true
        transitions = self.buffer.sample(self.args.batch_size)

        # pre-process the observation and goal
        o, o_next, g, ag, ag_next, actions, rewards = transitions['obs'], transitions['obs_next'], transitions['g'], transitions['ag'], \
                                                      transitions['ag_next'], transitions['actions'], transitions['r']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        _, transitions['ag'] = self._preproc_og(o, ag)
        _, transitions['ag_next'] = self._preproc_og(o, ag_next)

        # apply normalization
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        ag_norm = self.g_norm.normalize(transitions['ag'])
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])

        update_ddpg(self.model, self.policy_optim, self.critic_optim, obs_norm, ag_norm, g_norm, obs_next_norm, ag_next_norm, actions,
                        rewards, self.args)

    def save(self, model_path, epoch):
        torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                    self.model.actor.state_dict(), self.model.critic.state_dict()],
                    model_path + '/model_{}.pt'.format(epoch))
        
        # Save discovered goals to compute entropy after training
        torch.save(self.goal_sampler.discovered_goals, model_path + '/discovered_goals.pt')

    def load(self, model_path, args):

        o_mean, o_std, g_mean, g_std, actor, critic = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.actor.load_state_dict(actor)
        self.model.critic.load_state_dict(critic)
        self.model.actor.eval()
        self.o_norm.mean = o_mean
        self.o_norm.std = o_std
        self.g_norm.mean = g_mean
        self.g_norm.std = g_std
