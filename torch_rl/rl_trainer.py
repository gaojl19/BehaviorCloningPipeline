from utils.utils import *
from metaworld_utils.meta_env import generate_single_mt_env
from torch_rl.replay_buffer import EnvInfo
from policy.continuous_policy import MultiHeadGuassianContPolicy, EmbeddingGuassianContPolicyBase
from utils.utils import Path
from agents.bc_agent import SoftModuleAgent
from torch_rl.multi_task_collector import SingleCollector, MT10SingleCollector, MT50SingleCollector, MTEnvCollector

import torch
import numpy as np
import os
import time
from collections import OrderedDict
import seaborn as sns
import pandas as pd
import torch.multiprocessing as mp
from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT, EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, HARD_MODE_ARGS_KWARGS
import time

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):
    def __init__(self, env, env_cls, env_args, args, params, expert_dict, example_embedding = None):
        
        # environment
        self.env = env
        self.env_cls = env_cls
        self.env_args = env_args
        self.env_info = EnvInfo(
            env, params['general_setting']['device'], params['general_setting']['train_render'], params['general_setting']['eval_render'],
            params['general_setting']['epoch_frames'], params['general_setting']['eval_episodes'],
            params['general_setting']['max_episode_frames'], True, None
        )
        self.env_name = params["env_name"]
        
        
        # Agent
        self.args = args
        self.params = params
        
        agent_class = self.args['agent_class']
        if agent_class == SoftModuleAgent:
            self.agent = agent_class(self.env, example_embedding, self.args['agent_params'], self.params)
        else:
            self.agent = agent_class(self.env, self.args['agent_params'])
        
        self.input_shape = self.agent.actor.input_shape
        
        # MT10
        if self.params["env_name"] == "mt10":
            self.expert_env = MT10SingleCollector(
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                expert_dict=expert_dict,
                device=params['general_setting']['device'],
                max_path_length=self.args["ep_len"],
                min_timesteps_per_batch=self.args['batch_size'],
                params=params
            )
            self.agent_env = MTEnvCollector(
                env=self.env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                args=args,
                params=params,
                example_embedding=example_embedding
            )
            self.mt_flag = True
        
        elif self.params["env_name"] == "mt50":
            self.expert_env = MT50SingleCollector(
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                expert_dict=expert_dict,
                device=params['general_setting']['device'],
                max_path_length=self.args["ep_len"],
                min_timesteps_per_batch=self.args['batch_size'],
                params=params
            )
            self.agent_env = MTEnvCollector(
                env=self.env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                args=args,
                params=params,
                example_embedding=example_embedding
            )
            self.mt_flag = True
        
        else:
            self.expert_env = SingleCollector(
                env=env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                expert_policy=expert_dict[self.args["task_name"]],
                device=params['general_setting']['device'],
                max_path_length=self.args["ep_len"],
                min_timesteps_per_batch=self.args['batch_size'],
                input_shape = self.input_shape
            )
            
            self.mt_flag=False
            
        
        # build log dir
        plot_prefix = "./fig/"+ self.env_name
        if self.params["meta_env"]["random_init"] == False:
            plot_prefix += "_fixed/"
        else:
            plot_prefix += "_random/"
        
        if not os.path.isdir(plot_prefix):
            os.makedirs(plot_prefix)
        
        if self.mt_flag == False:
            plot_prefix += self.args["task_name"] + "_"
        self.plot_prefix = plot_prefix
    
    
    def run_training_loop(self, n_iter, relabel_with_expert=False, expert_task_curve={}, agent_task_curve={}):
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
    
        # TRAIN
        for itr in range(n_iter):
            
            start = time.time()
            train_start_time = time.time()
            
            # collect trajectories, to be used for training
            if itr == 0:
                print("\n\n-------------------------------- Iteration %i -------------------------------- "%itr)
                render = self.params["general_setting"]["train_render"] if (itr % self.args["render_interval"] == 0) else False
                training_returns = self.expert_env.sample_expert(render=render, render_mode="rgb_array", log=True, log_prefix = self.plot_prefix)
                # training_returns = self.collect_training_trajectories(
                #     itr,
                #     env_name,
                #     expert_policy,
                #     collect_policy,
                #     self.args['batch_size'],
                #     relabel_with_expert
                # )  # HW1: implement this function below
            
                paths, envsteps_this_batch, infos = training_returns
                self.total_envsteps += envsteps_this_batch

                # DAgger: relabel the collected obs with actions from a provided expert policy
                # if relabel_with_expert and itr>=start_relabel_with_expert:
                #     paths = self.do_relabel_with_expert(expert_policy, paths)

                # add collected data to replay buffer
                self.agent.add_to_replay_buffer(paths)

            
            # train agent (using sampled data from replay buffer)
            training_logs = self.train_agent()  # HW1: implement this function below
            
            min_loss = 1000
            for log in training_logs:
                loss_curve.append(log["Training Loss"])
                if min_loss > log["Training Loss"]:
                    min_loss = log["Training Loss"]
            
            # single_task
            if self.mt_flag == False:
                for path in paths:
                    expert_success_curve.append(path["success"])
            
            # multi-task
            else:
                for name in infos.keys():
                    if name =="mean_success_rate":
                        expert_success_curve.append(infos["mean_success_rate"])
                    else:
                        expert_task_curve[name].append(infos[name])

            train_time = time.time() - train_start_time
            
            eval_start_time = time.time()

            # EVALUATION
            if itr % self.args["eval_interval"] == 0:
                print("\n\n-------------------------------- Iteration %i -------------------------------- "%itr)
                render = False
                if self.mt_flag == False:
                    eval_success_rate = self.expert_env.sample_agent(agent_policy=self.agent.actor, n_sample=self.params["general_setting"]["eval_episodes"], render=render, render_mode="rgb_array", log=True, log_prefix = self.plot_prefix)
                    agent_success_curve.append(eval_success_rate)
                else:
                    eval_infos = self.agent_env.sample_agent(log_prefix=self.plot_prefix, agent_policy=self.agent.actor.policy, input_shape = self.agent.actor.input_shape, render=render)
                    for name in eval_infos.keys():
                        if name == "mean_success_rate":
                            agent_success_curve.append(eval_infos["mean_success_rate"])
                        else:
                            agent_task_curve[name].append(eval_infos[name])

                eval_time = time.time() - eval_start_time
                print("training time: ", train_time)
                print("evaluation time: ", eval_time)
                print("epoch time: ", time.time() - start)
            
            if min_loss < 0.1:
                print("\n\n-------------------------------- Training stopped due to early stopping -------------------------------- ")
                print("min loss: ", min_loss)
                break
        
        # TEST
        print("\n\n-------------------------------- Test Results -------------------------------- ")
        render = False
        if self.mt_flag == False:
            eval_success_rate = self.expert_env.sample_agent(agent_policy=self.agent.actor, n_sample=self.params["general_setting"]["eval_episodes"], render=render, render_mode="rgb_array", log=True, log_prefix = self.plot_prefix)
            print("mean_success_rate: ", eval_success_rate)
        else:
            eval_infos = self.agent_env.sample_agent(log_prefix=self.plot_prefix, agent_policy=self.agent.actor.policy, input_shape = self.agent.actor.input_shape, render=render)
            for name in eval_infos.keys():
                if name == "mean_success_rate":
                    continue
                else:
                    print(name, "_success_rate: ",  eval_infos[name])
            print("mean_success_rate: ", eval_infos["mean_success_rate"])
        
        # PLOT CURVE
        # plot overall loss curve
        iteration = range(len(loss_curve)-1)
        data = pd.DataFrame(loss_curve[1:], iteration)
        ax=sns.lineplot(data=data)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.set_title("BC--Loss Curve")
        
        fig = ax.get_figure()
        fig.savefig(self.plot_prefix + "Loss_lr_"+str(self.args['learning_rate']*10000)+".png")
        fig.clf()
        
        # plot expert success curve
        self.plot_success_curve(expert_success_curve, "expert", self.plot_prefix)
        
        # plot agent success curve
        self.plot_success_curve(agent_success_curve, "agent", self.plot_prefix)
        
        # for multi-task, plot single-task curve
        if self.mt_flag:
            # expert
            for task_name in expert_task_curve.keys():
                self.plot_single_curve(expert_task_curve[task_name], "expert", self.plot_prefix, task_name)
            
            # agent
            for task_name in agent_task_curve.keys():
                self.plot_single_curve(agent_task_curve[task_name], "agent", self.plot_prefix, task_name)
   
    def train_agent(self):
        all_logs = []
        for _ in range(self.args['gradient_steps']):

            # sample some data from the data buffer
            if self.mt_flag == True:
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch, embedding_batch = self.agent.sample(self.args['train_batch_size'])

                # use the sampled data to train an agent
                train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch, embedding_batch)

            else:
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.args['train_batch_size'])

                # use the sampled data to train an agent
                train_log = self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
            all_logs.append(train_log)
            
        return all_logs
            
    def plot_single_curve(self, curve, tag, plot_prefix, task_name):
        iteration = range(1, len(curve)+1)
        data = pd.DataFrame(curve, iteration)
        ax=sns.lineplot(data=data)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(tag + " success")
        ax.set_title(tag + "_" + task_name + " Mean Success Curve")
        ax.set(ylim=(-0.1, 1.1))
        
        fig = ax.get_figure()
        fig.savefig(plot_prefix + task_name + "_"+ tag + "_success_curve.png")
        
        fig.clf()
        
    def plot_success_curve(self, curve, tag, plot_prefix):
        iteration = range(1, len(curve)+1)
        data = pd.DataFrame(curve, iteration)
        ax=sns.lineplot(data=data)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(tag + " success")
        ax.set_title(tag + " Mean Success Curve")
        ax.set(ylim=(-0.1, 1.1))
        
        fig = ax.get_figure()
        fig.savefig(plot_prefix + tag + "_success_curve.png")
        
        fig.clf()
