import torch
import numpy as np
import torch.nn.functional as F
import time
from mpi_utils.mpi_utils import sync_grads


def update_ddpg(model, policy_optim, critic_optim, obs_norm, ag_norm, g_norm,
                    obs_next_norm, ag_next_norm, actions, rewards, args):
    # Tensorize
    obs_norm_tensor = torch.tensor(obs_norm, dtype=torch.float32)
    obs_next_norm_tensor = torch.tensor(obs_next_norm, dtype=torch.float32)
    g_norm_tensor = torch.tensor(g_norm, dtype=torch.float32)
    ag_norm_tensor = torch.tensor(ag_norm, dtype=torch.float32)
    ag_next_norm_tensor = torch.tensor(ag_next_norm, dtype=torch.float32)
    actions_tensor = torch.tensor(actions, dtype=torch.float32)
    r_tensor = torch.tensor(rewards, dtype=torch.float32).reshape(rewards.shape[0], 1)

    if args.cuda:
        obs_norm_tensor = obs_norm_tensor.cuda()
        obs_next_norm_tensor = obs_next_norm_tensor.cuda()
        g_norm_tensor = g_norm_tensor.cuda()
        ag_norm_tensor = ag_norm_tensor.cuda()
        ag_next_norm_tensor = ag_next_norm_tensor.cuda()
        actions_tensor = actions_tensor.cuda()
        r_tensor = r_tensor.cuda()

    with torch.no_grad():
        model.forward_pass(obs_next_norm_tensor, ag_next_norm_tensor, g_norm_tensor)
        qf1_next_target, qf2_next_target = model.target_q1_pi_tensor, model.target_q2_pi_tensor
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
        next_q_value = r_tensor + args.gamma * min_qf_next_target

    # the q loss
    qf1, qf2 = model.forward_pass(obs_norm_tensor, ag_norm_tensor, g_norm_tensor, actions=actions_tensor)
    qf1_loss = F.mse_loss(qf1, next_q_value)
    qf2_loss = F.mse_loss(qf2, next_q_value)
    qf_loss = qf1_loss + qf2_loss

    # the actor loss
    pi, log_pi = model.pi_tensor, model.log_prob
    qf1_pi, qf2_pi = model.q1_pi_tensor, model.q2_pi_tensor
    min_qf_pi = torch.min(qf1_pi, qf2_pi)
    # policy_loss = ((alpha * log_pi) - min_qf_pi).mean()
    policy_loss = -min_qf_pi.mean() + args.action_l2 * (pi/args.env_params['action_max']).pow(2).mean()

    # start to update the network
    policy_optim.zero_grad()
    policy_loss.backward(retain_graph=True)
    sync_grads(model.actor)
    policy_optim.step()

    # update the critic_network
    critic_optim.zero_grad()
    qf_loss.backward()
    sync_grads(model.critic)
    critic_optim.step()

    return qf1_loss.item(), qf2_loss.item(), policy_loss.item()
