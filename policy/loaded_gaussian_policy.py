from .base_policy import BasePolicy
from .continuous_policy import MultiHeadGuassianContPolicy
import torch
import numpy as np

class LoadedGaussianPolicy(BasePolicy):
    def __init__(self, env, params, args, **kwargs):
        super().__init__(**kwargs)

        # load policy networks
        self.pf = MultiHeadGuassianContPolicy (
        input_shape = env.observation_space.shape[0], 
        output_shape = 2 * env.action_space.shape[0],
        head_num=env.num_tasks,
        **params['net'] )
        
        self.pf.load_state_dict(torch.load(args["expert_policy_file"], map_location='cpu'))
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