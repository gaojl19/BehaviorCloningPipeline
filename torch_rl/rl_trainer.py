from utils.utils import *
from metaworld_utils.meta_env import generate_single_mt_env
from torch_rl.replay_buffer import EnvInfo

import torch
import numpy as np
import os
import time
from collections import OrderedDict
import seaborn as sns
import pandas as pd


# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):

    def __init__(self, env, env_cls, env_args, args, params):
        
        # environment
        self.env = env
        self.env_cls = env_cls
        self.env_args = env_args
        self.env_info = EnvInfo(
            env, params['general_setting']['device'], params['general_setting']['train_render'], params['general_setting']['eval_render'],
            params['general_setting']['epoch_frames'], params['general_setting']['eval_episodes'],
            params['general_setting']['max_episode_frames'], True, None
        )
        
        # Agent
        self.args = args
        self.params = params
        
        # logger
        self.logger = params['general_setting']['logger']
        
        # Set Max video length
        MAX_VIDEO_LEN = self.args['ep_len']
        
        agent_class = self.args['agent_class']
        self.agent = agent_class(self.env, self.args['agent_params'])
        print(self.agent.actor)

    
    def run_training_loop(self, n_iter, collect_policy, eval_policy,
                        initial_expertdata=None, relabel_with_expert=False,
                        start_relabel_with_expert=1, expert_policy=None):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        loss_curve = []
        expert_success_curve = []
        agent_success_curve = []

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************"%itr)

            # decide if videos should be rendered/logged at this iteration
            if itr % self.args['video_log_freq'] == 0 and self.args['video_log_freq'] != -1:
                self.log_video = True
            else:
                self.log_video = False

            # decide if metrics should be logged
            if itr % self.args['scalar_log_freq'] == 0:
                self.log_metrics = True
            else:
                self.log_metrics = False

            # Rebuild Env: only support single task env
            # TODO: if you want to upgrade this to multi-task, you can reference to async_mt.start_worker()
            self.env_info.env_cls = generate_single_mt_env
        
            tasks = list(self.env_cls.keys())
            single_mt_env_args = {
                "task_cls": None,
                "task_args": None,
                "env_rank": 0,
                "num_tasks": 1,
                "max_obs_dim": np.prod(self.env.observation_space.shape),
                "env_params": self.env_args[0],
                "meta_env_params": self.env_args[2]
            }

            for i, task in enumerate(tasks): # currently only 1 task
                env_cls = self.env_cls[task]
                
                self.env_info.env_rank = i 
    
                self.env_info.env_args = single_mt_env_args
                self.env_info.env_args["task_cls"] = env_cls
                self.env_info.env_args["task_args"] = copy.deepcopy(self.env_args[1][task])

                self.env_info.env_args["env_rank"] = i
                self.env_info.env = self.env_info.env_cls(**self.env_info.env_args)
               
            print(self.env_info.env)
            self.env_info.env.eval()
            print(self.env_info.env_args)
            
            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr,
                expert_policy,
                collect_policy,
                self.args['batch_size'],
                relabel_with_expert
            )  # HW1: implement this function below
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # DAgger: relabel the collected obs with actions from a provided expert policy
            if relabel_with_expert and itr>=start_relabel_with_expert:
                paths = self.do_relabel_with_expert(expert_policy, paths)  # HW1: implement this function below

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            training_logs = self.train_agent()  # HW1: implement this function below
            
            for log in training_logs:
                loss_curve.append(log["Training Loss"])
                
            for path in paths:
                expert_success_curve.append(path["success"])

            # log/save
            if self.log_video or self.log_metrics:

                # perform logging
                print('\nBeginning logging procedure...')
                success = self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, training_logs)
                agent_success_curve += success

                if self.args['save_params']:
                    print('\nSaving agent params')
                    self.agent.save('{}/policy_itr_{}.pt'.format(self.args['log_dir'], itr))
        
        # plot overall loss curve
        sns.set_theme(style="darkgrid")
        sns.set_palette(palette="gist_earth")

        iteration = range(len(loss_curve)-1)
        print(loss_curve[0])
        data = pd.DataFrame(loss_curve[1:], iteration)
        ax=sns.lineplot(data=data)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("BC--Loss Curve")
        
        fig = ax.get_figure()
        if not os.path.isdir("./fig"):
            os.makedirs("./fig")
        fig.savefig("./fig/BC_Loss_lr_"+str(self.args['learning_rate']*10000)+".png")
        fig.clf()
        
        # plot overall loss curve
        sns.set_theme(style="darkgrid")
        sns.set_palette(palette="gist_earth")

        iteration = range(len(loss_curve)-1)
        print(loss_curve[0])
        data = pd.DataFrame(loss_curve[1:], iteration)
        ax=sns.lineplot(data=data)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("BC--Loss Curve")
        
        fig = ax.get_figure()
        if not os.path.isdir("./fig"):
            os.makedirs("./fig")
        fig.savefig("./fig/BC_Loss_lr_"+str(self.args['learning_rate']*10000)+".png")
        fig.clf()
        
        print(expert_success_curve)
        print(agent_success_curve)
        
        # plot expert success curve
        iteration = range(1, len(expert_success_curve)+1)
        data = pd.DataFrame(expert_success_curve, iteration)
        ax=sns.lineplot(data=data)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("expert success")
        ax.set_title("BC--Expert Success Curve")
        
        fig = ax.get_figure()
        if not os.path.isdir("./fig"):
            os.makedirs("./fig")
        fig.savefig("./fig/" + self.args["task_name"] + "_expert_success_curve.png")
        fig.clf()

        # plot agent success curve
        iteration = range(1,len(agent_success_curve)+1)
        data = pd.DataFrame(agent_success_curve, iteration)
        ax=sns.lineplot(data=data)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("agent success")
        ax.set_title("BC--Agent Success Curve")
        
        fig = ax.get_figure()
        if not os.path.isdir("./fig"):
            os.makedirs("./fig")
        fig.savefig("./fig/" + self.args["task_name"] + "_agent_success_curve.png")
        fig.clf()
 
    def collect_training_trajectories(
            self,
            itr,
            expert_policy,
            collect_policy,
            batch_size,
            do_dagger,
    ):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """

        # TODO decide whether to load training data or use the current policy to collect more data
            
        # TODO collect `batch_size` samples to be used for training
        # HINT1: use sample_trajectories from utils
        # HINT2: you want each of these collected rollouts to be of length self.params['ep_len']
        print("\nCollecting data to be used for training...")
        
        if do_dagger:
            if itr == 0:
                # We use expert_policy to generate training data
                paths, envsteps_this_batch = sample_trajectories(self.env_info.env, expert_policy, self.args["device"], 
                                                         batch_size, self.args['ep_len'], render=True)
            else:
                # use agent to collect data, and then corrected by the expert policy later on
                paths, envsteps_this_batch = sample_trajectories(self.env_info.env, collect_policy, self.args["device"], 
                                                         batch_size, self.args['ep_len'], render=True, run_agent = True)
        else:
            # We use expert_policy to generate training data
            paths, envsteps_this_batch = sample_trajectories(self.env_info.env, expert_policy, self.args["device"], 
                                                            batch_size, self.args['ep_len'], render=True)
        

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        # note: here, we collect MAX_NVIDEO rollouts, each of length MAX_VIDEO_LEN
        train_video_paths = None
        if self.log_video:
            print('\nCollecting train rollouts to be used for saving videos...')
            ## TODO look in utils and implement sample_n_trajectories
            train_video_paths = sample_n_trajectories(self.env_info.env, collect_policy, MAX_NVIDEO, self.args["device"], MAX_VIDEO_LEN, run_agent = True)

        return paths, envsteps_this_batch, train_video_paths


    def train_agent(self):
        print('\nTraining agent using sampled data from replay buffer...')
        all_logs = []
        for train_step in range(self.args['num_agent_train_steps_per_iter']):

            # TODO sample some data from the data buffer
            # HINT1: use the agent's sample function
            # HINT2: how much data = self.params['train_batch_size']
            ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.args['train_batch_size'])

            # TODO use the sampled data to train an agent
            # HINT: use the agent's train function
            # HINT: keep the agent's training log for debugging
            train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
            
        return all_logs

    # for DAgger
    def do_relabel_with_expert(self, expert_policy, paths):
        print("\nRelabelling collected observations with labels from an expert policy...")

        # TODO relabel collected obsevations (from our policy) with labels from an expert policy
        # HINT: query the policy (using the get_action function) with paths[i]["observation"]
        # and replace paths[i]["action"] with these expert labels
        for i in range(len(paths)):
            new_acts = []
            for ob in paths[i]["observation"]:
                expert_act= expert_policy.get_action(torch.Tensor(ob), self.args["device"])
                new_acts.append(expert_act)
            paths[i]["action"] = new_acts
            
        return paths


    def perform_logging(self, itr, paths, eval_policy, train_video_paths, training_logs):

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_policy.eval() # frozen before
        eval_paths, eval_envsteps_this_batch = sample_trajectories(self.env_info.env, eval_policy, self.args["device"], self.args['eval_batch_size'], self.args['ep_len'], run_agent=True)

        
        # save eval rollouts as videos in tensorboard event file
        if self.log_video and train_video_paths != None:
            print('\nCollecting video rollouts eval')
            eval_video_paths = sample_n_trajectories(self.env_info.env, eval_policy, MAX_NVIDEO, MAX_VIDEO_LEN, True)

            #save train/eval videos
            print('\nSaving train rollouts as videos...')
            self.logger.log_paths_as_videos(train_video_paths, itr, fps=self.fps, max_videos_to_save=MAX_NVIDEO,
                                            video_title='train_rollouts')
            self.logger.log_paths_as_videos(eval_video_paths, itr, fps=self.fps,max_videos_to_save=MAX_NVIDEO,
                                             video_title='eval_rollouts')

        # save eval metrics
        if self.log_metrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            last_log = training_logs[-1]  # Only use the last log for now
            logs.update(last_log)


            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print('{} : {}'.format(key, value))
                self.logger.log_scalar(value, key, itr)
            print('Done logging...\n\n')

            self.logger.flush()
            
        success = []
        for path in eval_paths:
            success.append(path["success"])
        return success
