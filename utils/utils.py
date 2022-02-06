import numpy as np
import torch
import time
import copy
import imageio


def Path(obs, image_obs, acs, rewards, next_obs, terminals, success, embedding_input = [], index_input = []):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
        
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32),
            "success": np.array(success, dtype=np.float32),
            "embedding_input": embedding_input,
            "index_input": index_input}


def convert_listofrollouts(paths, concat_rew=True, embedding_flag=False, index_flag=False):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """

    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    if concat_rew:
        rewards = np.concatenate([path["reward"] for path in paths])
    else:
        rewards = [path["reward"] for path in paths]
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    
    
    if embedding_flag:
        embedding_input= np.concatenate([path["embedding_input"] for path in paths])
        return observations, actions, rewards, next_observations, terminals, embedding_input

    elif index_flag:
        index_input = np.concatenate([path["index_input"] for path in paths])
        return observations, actions, rewards, next_observations, terminals, index_input
    # single task doesn't need task embedding
    else:
        return observations, actions, rewards, next_observations, terminals


def get_pathlength(path):
    return len(path["reward"])