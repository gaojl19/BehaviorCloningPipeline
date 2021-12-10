import numpy as np
import torch
import time
import copy
import imageio

############################################
############################################

# the policy gradient should be frozen before sending into this function
def sample_trajectory(env, policy, device, max_path_length, render=False, render_mode=('rgb_array'), run_agent=False):
    
    # initialize env for the beginning of a new rollout
    env.eval()
    ob = env.reset()

    
    # init vars
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0
    done = False
    success = 0
    while True:
        
        # use the most recent ob to decide what to do
        obs.append(ob)
        
        # query the policy's get_action function
        if not run_agent:
            act = policy.get_action(torch.Tensor(ob).to(device).unsqueeze(0), device)
            print("expert:",act)
        
        else:
            act = policy.get_action(torch.Tensor(ob).to(device).unsqueeze(0)).detach().cpu().numpy()
            act = np.squeeze(act)
            print("imitation agent:", act)

        # print(act)
        acs.append(act)
        
        # take that action and record results
        ob, r, done, info = env.step(act)
        
        # record result of taking that action
        steps += 1
        next_obs.append(ob)
        rewards.append(r)
        
        # only support rbg_array mode currently
        if render:
            # image_obs.append(env.render(mode='rgb_array'))
            image = env.get_image(400,400,None)
            image_obs.append(image)
        
        success = max(success, info["success"])

        # end the rollout if the rollout ended
        rollout_done = True if (done or steps>=max_path_length) else False
        terminals.append(rollout_done)

        if rollout_done:
            break
        
    print("success: ", success)
    if len(image_obs)>0:
        imageio.mimsave("expert.gif", image_obs)

    return Path(obs, image_obs, acs, rewards, next_obs, terminals, success)


def sample_trajectories(env, policy, device, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array'), run_agent=False):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.
        
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        new_path = sample_trajectory(env=env, policy=policy, device=device, max_path_length=max_path_length, render=render, render_mode=render_mode, run_agent=run_agent)
        paths.append(new_path)
        timesteps_this_batch += get_pathlength(new_path)

    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, device, max_path_length, render=False, render_mode=('rgb_array'), run_agent=False):
    """
        Collect ntraj rollouts.
        
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    paths = []

    for i in range(ntraj):
        paths.append(sample_trajectory(env=env, policy=policy, device=device, max_path_length=max_path_length, render=render, render_mode=render_mode, run_agent=run_agent))

    return paths


def Path(obs, image_obs, acs, rewards, next_obs, terminals, success):
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
            "success":np.array(success, dtype=np.float32)}


def convert_listofrollouts(paths, concat_rew=True):
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
    return observations, actions, rewards, next_observations, terminals


def get_pathlength(path):
    return len(path["reward"])