from .base_policy import BasePolicy
from .continuous_policy import MultiHeadGuassianContPolicy
from .continuous_policy import ModularGuassianGatedCascadeCondContPolicy
import torch
import numpy as np


class LoadedGaussianPolicy(BasePolicy):
    def __init__(self, env, params, policy_path, **kwargs):
        super().__init__(**kwargs)

        # load policy networks
        if params["env_name"] == "single_task":
            self.pf = MultiHeadGuassianContPolicy (
            input_shape = env.observation_space.shape[0], 
            output_shape = 2 * env.action_space.shape[0],
            head_num=env.num_tasks,
            **params['net'] )
        else:
            self.pf = MultiHeadGuassianContPolicy (
            input_shape = env.observation_space.shape[0], 
            output_shape = 2 * env.action_space.shape[0],
            head_num = 1,
            **params['expert_net'] )
        
        self.pf.load_state_dict(torch.load(policy_path, map_location='cpu'))
        self.pf.eval()

    def forward(self, obs, idx=torch.LongTensor([0])):
        self.pf.forward(obs, idx)


    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        raise NotImplementedError("""
            This policy class simply loads in a particular type of policy and
            queries it. Do not try to train it.
        """)

    # for single task: idx=0
    def get_action(self, obs, device, idx=torch.LongTensor([0])):
        obs = obs.detach()
            
        idx = torch.Tensor([[idx]]).to(device).long()
        return self.pf.eval_act(obs, idx)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)
        
        

class LoadedModularGuassianGatedCascadeCondContPolicy(BasePolicy):
    def __init__(self, env, params, args, example_embedding, **kwargs):
        super().__init__(**kwargs)

        # load policy networks
        self.pf = ModularGuassianGatedCascadeCondContPolicy(
        input_shape=env.observation_space.shape[0],
        em_input_shape=np.prod(example_embedding.shape),
        output_shape=2 * env.action_space.shape[0],
        **params['net'])
        
        self.pf.load_state_dict(torch.load(args["expert_policy_file"], map_location='cpu'))
        self.pf.eval()

    def forward(self, obs, idx):
        self.pf.forward(obs, idx)

    def update(self, obs_no, acs_na, adv_n=None, acs_labels_na=None):
        raise NotImplementedError("""
            This policy class simply loads in a particular type of policy and
            queries it. Do not try to train it.
        """)

    # for single task: idx=0
    def get_action(self, obs, device, idx):
        obs = obs.detach()
            
        idx = torch.Tensor(idx).to(device).long()
        return self.pf.eval_act(obs, idx)

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)