from torch_rl.replay_buffer import ReplayBuffer
from policy.MLP_policy import MLPPolicy
from policy.Soft_Module_policy import SoftModulePolicy
from .base_agent import BaseAgent
import torch.nn as nn
import torch.optim as optim


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

        # update
        self.loss = nn.MSELoss()
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
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
        return self.replay_buffer.sample_mt_random_data(batch_size) 

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
        
        # update
        self.loss = nn.MSELoss()
        self.learning_rate = self.agent_params['learning_rate']
        self.optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate,
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(self.agent_params['max_replay_buffer_size'])

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n, embedding_input_n):
        self.actor.train()
        
        self.optimizer.zero_grad()
        pred_acs = self.actor(ob_no, embedding_input_n.squeeze())
        loss = self.loss(pred_acs, ac_na)

        loss.backward()
        self.optimizer.step()
        
        log = {
            # You can add extra logging information here, but keep this line
            'Training Loss': loss.to('cpu').detach().numpy(),
        } 
        return log

    def add_mt_to_replay_buffer(self, paths):
        self.replay_buffer.add_mt_rollouts(paths)

    def mt_sample(self, batch_size):
        return self.replay_buffer.sample_mt_random_data(batch_size)  # HW1: you will modify this

    def save(self, path):
        return self.actor.save(path)