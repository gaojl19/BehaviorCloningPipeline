from shutil import Error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import networks.init as init
from .continuous_policy import ModularGuassianGatedCascadeCondContPolicy
from .distribution import TanhNormal

class SoftModulePolicy(nn.Module):
    def __init__(self, env, example_embedding, params):
        super().__init__()
        self.policy = ModularGuassianGatedCascadeCondContPolicy(
        input_shape=env.observation_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=2 * env.action_space.shape[0],
        **params['net']
        )
        
    def forward(self, x, embedding_input = None):
        if embedding_input == None:
            raise Error
        mean, std, log_std = self.policy.forward(x, embedding_input)
        dis = TanhNormal(mean, std)
        action = dis.rsample( return_pretanh_value = False )
        return action
    
    
    def get_action(self, obs: np.ndarray, embedding_input):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        return self.forward(observation, embedding_input)