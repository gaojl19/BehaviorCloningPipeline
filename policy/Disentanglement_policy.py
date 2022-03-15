import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import networks.init as init


class DisentanglementPolicy(nn.Module):
    def __init__(self, state_shape, output_shape, hidden_shape, example_embedding, activation_func=F.relu, init_func = init.basic_init, last_activation_func = None ):
        super().__init__()
        
        self.activation_func = activation_func
        if last_activation_func is not None:
            self.last_activation_func = last_activation_func
        else:
            self.last_activation_func = None
    
        
        # state embedding
        # self.embedding1 = nn.Linear(state_shape, hidden_shape)
        # self.embedding2 = nn.Linear(example_embedding.shape[0], hidden_shape)
        # init_func(self.embedding1)
        # init_func(self.embedding2)
        # self.__setattr__("embedding1", self.embedding1)
        # self.__setattr__("embedding2", self.embedding2)
        self.input_shape = state_shape
        
        self.embedding = nn.Linear(state_shape+example_embedding.shape[0], hidden_shape)
        init_func(self.embedding)
        self.__setattr__("embedding", self.embedding)
        
        
        # encoder
        fc = nn.Linear(hidden_shape, example_embedding.shape[0])
        init_func(fc)
        self.encoder = fc
        # set attr for pytorch to track parameters( device )
        self.__setattr__("encoder", fc)
        
        # decoder
        fc = nn.Linear(example_embedding.shape[0], output_shape)
        init_func(fc)
        self.decoder = fc
        self.__setattr__("decoder", fc)
        
        
    
    def forward(self, x, embedding_input_n, return_feature = True):

        input = torch.Tensor(np.concatenate((x, embedding_input_n.squeeze(1)), axis=1))
        input = self.activation_func(self.embedding(input))
        
        feature = self.activation_func(self.encoder(input))
        out = self.activation_func(self.decoder(feature))
        
        if return_feature:
            return out, feature
        else:
            return out
    
    
    def get_action(self, obs: np.ndarray):
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # return the action that the policy prescribes
        return self.forward(observation)