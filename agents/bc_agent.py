from torch_rl.replay_buffer import ReplayBuffer
from policy.MLP_policy import MLPPolicy
from policy.Soft_Module_policy import SoftModulePolicy
from policy.MH_SAC_policy import MHSACPolicy
from policy.IQ_learn_policy import *
from .base_agent import BaseAgent
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
# import hydra


class MLPAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(MLPAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MLPPolicy(
            input_shape = self.agent_params['ob_dim'],
            output_shape = self.agent_params['ac_dim'],
            n_layers = self.agent_params['n_layers'],
            hidden_shape = self.agent_params['size']
        )
        
        print("actor: \n", self.actor)

        # update
        self.loss = nn.MSELoss()
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na):
        self.actor.train()
        
        self.optimizer.zero_grad()
        
        pred_acs = self.actor(ob_no)
        loss = self.loss(pred_acs, ac_na)

        loss.backward()
        self.optimizer.step()
        
        log = {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.to('cpu').detach().numpy(),
        } 
        return log

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)
        
    def add_mt_to_replay_buffer(self, paths):
        self.replay_buffer.add_mt_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size) 

    def mt_sample(self, batch_size):
        return self.replay_buffer.sample_random_data_embedding(batch_size) 

    def save(self, path):
        return self.actor.save(path)
    
    
    
class MLPEmbeddingAgent(BaseAgent):
    def __init__(self, env, example_embedding, agent_params, params):
        super(MLPEmbeddingAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        # add the embedding vector to the input
        self.actor = MLPPolicy(
            input_shape = self.agent_params['ob_dim'] + np.prod(example_embedding.shape),
            output_shape = self.agent_params['ac_dim'],
            n_layers = self.agent_params['n_layers'],
            hidden_shape = self.agent_params['size']
        )
        
        print("actor: \n", self.actor)

        # update
        self.loss = nn.MSELoss()
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, embedding_input_n, alternate=-1):
        self.actor.train()
        
        self.optimizer.zero_grad()
        input = torch.Tensor(np.concatenate((ob_no, embedding_input_n.squeeze()), axis=1))
        
        pred_acs = self.actor(input)
        loss = self.loss(pred_acs, ac_na)

        loss.backward()
        self.optimizer.step()
        
        log = {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.to('cpu').detach().numpy(),
        } 
        
        return log

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)
        
    def add_mt_to_replay_buffer(self, paths):
        self.replay_buffer.add_embedding_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_random_data(batch_size) 

    def mt_sample(self, batch_size):
        return self.replay_buffer.sample_random_data_embedding(batch_size) 

    def save(self, path):
        return self.actor.save(path)


class SoftModuleAgent(BaseAgent):
    def __init__(self, env, example_embedding, agent_params, params):
        super(SoftModuleAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = SoftModulePolicy(
            env = env,
            example_embedding = example_embedding,
            params = params
        )
        
        print("actor: \n", self.actor.policy)
        
        # update
        self.loss = nn.MSELoss()
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.policy.parameters(),
            lr=self.learning_rate,
        )
        
        # define 2 optimizers for training 2 parts of networks
        p1 = []
        p2 = []
        for name, p in self.actor.policy.named_parameters():
            if "gating" in name or "em_base" in name:
                p2.append(p)
            else:
                p1.append(p)
                
        self.optimizer1 = optim.Adam(
            p1,
            lr=self.learning_rate
        )
        self.optimizer2 = optim.Adam(
            p2,
            lr=self.learning_rate
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, embedding_input_n, alternate=-1):
        self.actor.train()
        pred_acs, weights = self.actor(ob_no, embedding_input_n.squeeze())
        loss = self.loss(pred_acs, ac_na)
        
        if alternate == 0:  # train base policy
            print("train policy!\n")
            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()

        
        elif alternate == 1: # train routing network
            print("train routing!\n")
            self.optimizer2.zero_grad()
            loss.backward()
            self.optimizer2.step()
            
        else:   # train full net
            self.optimizer.zero_grad()

            # add L1 regularization
            if self.agent_params["l1_regularization"]:
                l1_lambda = self.agent_params["l1_lambda"]  # 0.001
                # l1_lambda = 1e-7
                l1_norm = 0.0
                
                # separate routing networks from the whole policy networks
                # includes:
                #   gating_fc_0.weight
                #   gating_fc_0.bias
                #   gating_fc_1.weight
                #   gating_fc_1.bias
                #   gating_weight_fc_0.weight
                #   gating_weight_fc_0.bias
                #   gating_weight_cond_last.weight
                #   gating_weight_cond_last.bias
                #   gating_weight_last.weight
                #   gating_weight_last.bias
                for name, p in self.actor.policy.named_parameters():
                    if "gating" in name and "bias" not in name:
                        # print(name)
                        l1_norm += p.abs().sum() 
                        # print(name, ": max->", p.abs().max(), "sum->", p.abs().sum()) 
                # print("l1 norm:", l1_norm)
                # print("prediction loss: ", loss)
                loss = loss + l1_lambda * l1_norm
            
            elif self.agent_params["regularize_weights"]:
                l1_lambda = self.agent_params["l1_lambda"]  # 0.001
                l1_norm = 0
                for w in weights:
                    l1_norm += w.abs().sum()

                loss = loss + l1_lambda * l1_norm
                print(l1_lambda*l1_norm)

            loss.backward()
            
            # for name, p in self.actor.policy.named_parameters():
            #     if "gating" in name and "bias" not in name:
            #         print(p.grad)
            #         print(p)
            #         print(name, ": max->", p.abs().max(), "sum->", p.abs().sum()) 
            
            self.optimizer.step()

        log = {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.to('cpu').detach().numpy(),
        } 
        return log

    def add_mt_to_replay_buffer(self, paths):
        self.replay_buffer .add_embedding_rollouts(paths)

    def mt_sample(self, batch_size):
        return self.replay_buffer.sample_random_data_embedding(batch_size) 

    def save(self, path):
        return self.actor.save(path)
    
    
    
class MHSACAgent(BaseAgent):
    def __init__(self, env, agent_params, params):
        super(MHSACAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params

        # actor/policy
        self.actor = MHSACPolicy(
            env = env,
            params = params
        )
        
        print("actor: \n", self.actor.policy)
        
        # update
        self.loss = nn.MSELoss()
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, index_input_n):
        self.actor.train()
        
        self.optimizer.zero_grad()
        pred_acs = self.actor(ob_no, index_input_n.squeeze())
        loss = self.loss(pred_acs, ac_na)

        loss.backward()
        self.optimizer.step()
        
        log = {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.to('cpu').detach().numpy(),
        } 
        return log

    def add_mt_to_replay_buffer(self, paths):
        self.replay_buffer.add_index_rollouts(paths)

    def mt_sample(self, batch_size):
        return self.replay_buffer.sample_random_data_index(batch_size)

    def save(self, path):
        return self.actor.save(path)
    
    


# class IQLearnAgent(BaseAgent):
#     def __init__(self, obs_dim, action_dim, action_range, batch_size, args):
#         self.gamma = args.gamma
#         self.batch_size = batch_size
#         self.action_range = action_range
#         self.device = torch.device(args.device)
#         self.args = args
#         agent_cfg = args.agent

#         self.critic_tau = agent_cfg.critic_tau
#         self.learnable_temperature = agent_cfg.learnable_temperature
#         self.actor_update_frequency = agent_cfg.actor_update_frequency
#         self.critic_target_update_frequency = agent_cfg.critic_target_update_frequency

#         self.critic = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(self.device)

#         self.critic_target = hydra.utils.instantiate(agent_cfg.critic_cfg, args=args).to(
#             self.device)
#         self.critic_target.load_state_dict(self.critic.state_dict())

#         self.actor = hydra.utils.instantiate(agent_cfg.actor_cfg).to(self.device)
#         print("actor: \n", self.actor)

#         self.log_alpha = torch.tensor(np.log(agent_cfg.init_temp)).to(self.device)
#         self.log_alpha.requires_grad = True
#         # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
#         self.target_entropy = -action_dim

#         # optimizers
#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
#                                                 lr=agent_cfg.actor_lr,
#                                                 betas=agent_cfg.actor_betas)
#         self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=agent_cfg.critic_lr,
#                                      betas=agent_cfg.critic_betas)
#         self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
#                                                     lr=agent_cfg.alpha_lr,
#                                                     betas=agent_cfg.alpha_betas)
        
#         # replay buffer
#         self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])
        
#         # Train
#         self.train()
#         self.critic_target.train()

#     def train(self, training=True):
#         self.training = training
#         self.actor.train(training)
#         self.critic.train(training)
#         log = {
#             # You can add extra logging information here, but keep this line
#             'Training Loss': loss.to('cpu').detach().numpy(),
#         } 
#         return log

#     @property
#     def alpha(self):
#         return self.log_alpha.exp()

#     @property
#     def critic_net(self):
#         return self.critic

#     @property
#     def critic_target_net(self):
#         return self.critic_target

#     def choose_action(self, state, sample=False):
#         state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
#         dist = self.actor(state)
#         action = dist.sample() if sample else dist.mean
#         action = action.clamp(*self.action_range)
#         # assert action.ndim == 2 and action.shape[0] == 1
#         return action.detach().cpu().numpy()[0]

#     def sample_actions(self, obs, num_actions):
#         """For CQL style training"""
#         obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(
#             obs.shape[0] * num_actions, obs.shape[1])
#         action, log_prob, _ = self.actor.sample(obs_temp)
#         return action, log_prob.view(obs.shape[0], num_actions, 1)

#     def _get_tensor_values(self, obs, actions, network=None):
#         """For CQL style training"""
#         action_shape = actions.shape[0]
#         obs_shape = obs.shape[0]
#         num_repeat = int(action_shape / obs_shape)
#         obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(
#             obs.shape[0] * num_repeat, obs.shape[1])
#         preds = network(obs_temp, actions)
#         preds = preds.view(obs.shape[0], num_repeat, 1)
#         return preds

#     def cqlV(self, obs, network, num_random=10):
#         # importance sampled version
#         # action, log_prob, _ = self.actor.sample(obs)
#         action, log_prob = self.sample_actions(obs, num_random)
#         current_Q = self._get_tensor_values(obs, action, network)

#         random_action = torch.FloatTensor(
#             obs.shape[0] * num_random, action.shape[-1]).uniform_(-1, 1).to(self.device)

#         random_density = np.log(0.5 ** action.shape[-1])

#         rand_Q = self._get_tensor_values(obs, random_action, network)

#         alpha = self.alpha.detach()

#         cat_Q = torch.cat(
#             [rand_Q - alpha * random_density, current_Q - alpha * log_prob.detach()], 1
#         )

#         cql_V = torch.logsumexp(cat_Q / alpha, dim=1).mean() * alpha
#         return cql_V

#     def getV(self, obs):
#         action, log_prob, _ = self.actor.sample(obs)
#         current_Q = self.critic(obs, action)
#         current_V = current_Q - self.alpha.detach() * log_prob
#         return current_V

#     def get_targetV(self, obs):
#         action, log_prob, _ = self.actor.sample(obs)
#         target_Q = self.critic_target(obs, action)
#         target_V = target_Q - self.alpha.detach() * log_prob
#         return target_V

#     def update(self, replay_buffer, logger, step):
#         obs, next_obs, action, reward, done = replay_buffer.get_samples(
#             self.batch_size, self.device)

#         losses = self.update_critic(obs, action, reward, next_obs, done,
#                                     logger, step)

#         if step % self.actor_update_frequency == 0:
#             actor_alpha_losses = self.update_actor_and_alpha(obs, logger, step)
#             losses.update(actor_alpha_losses)

#         if step % self.critic_target_update_frequency == 0:
#             soft_update(self.critic, self.critic_target,
#                         self.critic_tau)

#         return losses

#     def update_critic(self, obs, action, reward, next_obs, done, logger,
#                       step):
#         # dist = self.actor(next_obs)
#         # next_action = dist.rsample()
#         # log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

#         with torch.no_grad():
#             next_action, log_prob, _ = self.actor.sample(next_obs)

#             target_Q = self.critic_target(next_obs, next_action)
#             target_V = target_Q - self.alpha.detach() * log_prob
#             target_Q = reward + (1 - done) * self.gamma * target_V

#         # get current Q estimates
#         current_Q1, current_Q2 = self.critic(obs, action, both=True)
#         q1_loss = F.mse_loss(current_Q1, target_Q)
#         q2_loss = F.mse_loss(current_Q2, target_Q)
#         critic_loss = q1_loss + q2_loss
#         # logger.log('train_critic/loss', critic_loss, step)

#         # Optimize the critic
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()

#         # self.critic.log(logger, step)
#         return {
#             'critic_loss/critic_1': q1_loss.item(),
#             'critic_loss/critic_2': q2_loss.item(),
#             'loss/critic': critic_loss.item()}

#     def update_actor_and_alpha(self, obs, logger, step):
#         # dist = self.actor(obs)
#         # action = dist.rsample()
#         # log_prob = dist.log_prob(action).sum(-1, keepdim=True)
#         # actor_Q1, actor_Q2 = self.critic(obs, action)

#         action, log_prob, _ = self.actor.sample(obs)
#         actor_Q = self.critic(obs, action)

#         actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

#         # logger.log('train_actor/loss', actor_loss, step)
#         # logger.log('train_actor/target_entropy', self.target_entropy, step)
#         # logger.log('train_actor/entropy', -log_prob.mean(), step)

#         # optimize the actor
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         losses = {
#             'loss/actor': actor_loss.item(),
#             'actor_loss/target_entropy': self.target_entropy,
#             'actor_loss/entropy': -log_prob.mean().item()}
#         # self.actor.log(logger, step)

#         if self.learnable_temperature:
#             self.log_alpha_optimizer.zero_grad()
#             alpha_loss = (self.alpha *
#                           (-log_prob - self.target_entropy).detach()).mean()
#             # logger.log('train_alpha/loss', alpha_loss, step)
#             # logger.log('train_alpha/value', self.alpha, step)
#             alpha_loss.backward()
#             self.log_alpha_optimizer.step()

#             losses.update({
#                 'alpha_loss/loss': alpha_loss.item(),
#                 'alpha_loss/value': self.alpha.item(),
#             })
#         return losses

#     # Save model parameters
#     def save(self, path, suffix=""):
#         actor_path = f"{path}{suffix}_actor"
#         critic_path = f"{path}{suffix}_critic"
#         # print('Saving models to {} and {}'.format(actor_path, critic_path))
#         torch.save(self.actor.state_dict(), actor_path)
#         torch.save(self.critic.state_dict(), critic_path)

#     # Load model parameters
#     def load(self, path, suffix=""):
#         actor_path = f'{path}/{self.args.agent.name}{suffix}_actor'
#         critic_path = f'{path}/{self.args.agent.name}{suffix}_critic'
#         print('Loading models from {} and {}'.format(actor_path, critic_path))
#         if actor_path is not None:
#             self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
#         if critic_path is not None:
#             self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

#     def infer_q(self, state, action):
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#         action = torch.FloatTensor(action).unsqueeze(0).to(self.device)

#         with torch.no_grad():
#             q = self.critic(state, action)
#         return q.squeeze(0).cpu().numpy()

#     def infer_v(self, state):
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#         with torch.no_grad():
#             v = self.getV(state).squeeze()
#         return v.cpu().numpy()
    
        
        
#         # # update
#         # self.loss = nn.MSELoss()
#         # self.learning_rate = self.agent_params['learning_rate']
#         # self.optimizer = optim.Adam(
#         #     self.actor.parameters(),
#         #     lr=self.learning_rate,
#         # )

#         # # replay buffer
#         # self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])
        

#     def add_mt_to_replay_buffer(self, paths):
#         self.replay_buffer.add_index_rollouts(paths)

#     def mt_sample(self, batch_size):
#         return self.replay_buffer.sample_random_data_index(batch_size)

#     def save(self, path):
#         return self.actor.save(path)