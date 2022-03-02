from utils.utils import *
from metaworld_utils.meta_env import generate_single_mt_env
from torch_rl.replay_buffer import EnvInfo
from policy.continuous_policy import MultiHeadGuassianContPolicy, EmbeddingGuassianContPolicyBase
from utils.utils import Path
from agents.bc_agent import MHSACAgent, MLPAgent, MLPEmbeddingAgent, SoftModuleAgent
from torch_rl.multi_task_collector import *

import torch
import numpy as np
import os
import time
from collections import OrderedDict
import seaborn as sns
import pandas as pd
import torch.multiprocessing as mp
from scipy.signal import savgol_filter
import time

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):
    def __init__(self, env, env_cls, env_args, args, params, expert_dict, input_shape, baseline=False, example_embedding = None):
        
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
        
        self.index_flag = False
        
        agent_class = self.args['agent_class']
        if agent_class == MLPAgent :
            self.agent = agent_class(self.env, self.args['agent_params'])
        elif agent_class == SoftModuleAgent or agent_class == MLPEmbeddingAgent:
            self.agent = agent_class(self.env, example_embedding, self.args['agent_params'], self.params)
        elif agent_class == MHSACAgent:
            self.agent = agent_class(self.env, self.args['agent_params'], self.params)
            self.index_flag = True
        else:
            raise NotImplementedError(agent_class)
            
        # expert has input shape of 9
        # agent will have input shape of 12(Augmented observation)
        self.input_shape = input_shape
        
        
        # build log dir
        plot_prefix = "./fig/"+ self.env_name
        if baseline:
            plot_prefix += "_baseline"
        if self.index_flag:
            plot_prefix += "_mhsac"
        if "num_layers" in self.params["net"].keys(): # for Soft-Module with different module architectures
            plot_prefix += "_"
            plot_prefix += str(self.params["net"]["num_layers"])
            plot_prefix += "_"
            plot_prefix += str(self.params["net"]["num_modules"])
            plot_prefix += "_"
            plot_prefix += str(self.params["net"]["module_hidden"])
        
        if "l1_regularization" in self.args.keys():
            if self.args["l1_regularization"]:
                plot_prefix += "_L1Norm"
        
        if "shared_base" in self.params["net"].keys():
            if self.params["net"]["shared_base"]:
                plot_prefix += "_sharedBase"
            
        if self.params["meta_env"]["random_init"] == False:
            plot_prefix += "_fixed/"
        else:
            plot_prefix += "_random/"
        
        if not os.path.isdir(plot_prefix):
            os.makedirs(plot_prefix)
        
        self.plot_prefix = plot_prefix
        print("plot prefix: ", plot_prefix)
        
        
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
                params=params,
                input_shape = self.input_shape
            )
            self.agent_env = MTEnvCollector(
                env=self.env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                args=args,
                params=params,
                example_embedding=example_embedding,
                plot_prefix=self.plot_prefix
            )
            self.mt_flag = True
        
        # MT10 diverse
        elif self.params["env_name"] == "mt10_diverse":
            self.expert_env = MT10DiverseCollector(
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                expert_dict=expert_dict,
                device=params['general_setting']['device'],
                max_path_length=self.args["ep_len"],
                min_timesteps_per_batch=self.args['batch_size'],
                params=params,
                input_shape = self.input_shape
            )
            self.agent_env = MTEnvCollector(
                env=self.env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                args=args,
                params=params,
                example_embedding=example_embedding,
                plot_prefix=self.plot_prefix
            )
            self.mt_flag = True
        
        # MT10 similar
        elif self.params["env_name"] == "mt10_similar":
            self.expert_env = MT10SimilarCollector(
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                expert_dict=expert_dict,
                device=params['general_setting']['device'],
                max_path_length=self.args["ep_len"],
                min_timesteps_per_batch=self.args['batch_size'],
                params=params,
                input_shape = self.input_shape
            )
            self.agent_env = MTEnvCollector(
                env=self.env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                args=args,
                params=params,
                example_embedding=example_embedding,
                plot_prefix=self.plot_prefix
            )
            self.mt_flag = True
        
        # MT10 fail
        elif self.params["env_name"] == "mt10_fail":
            self.expert_env = MT10FailCollector(
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                expert_dict=expert_dict,
                device=params['general_setting']['device'],
                max_path_length=self.args["ep_len"],
                min_timesteps_per_batch=self.args['batch_size'],
                params=params,
                input_shape = self.input_shape
            )
            self.agent_env = MTEnvCollector(
                env=self.env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                args=args,
                params=params,
                example_embedding=example_embedding,
                plot_prefix=self.plot_prefix
            )
            self.mt_flag = True
        
        # MT10 medium
        elif self.params["env_name"] == "mt10_medium":
            self.expert_env = MT10MediumCollector(
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                expert_dict=expert_dict,
                device=params['general_setting']['device'],
                max_path_length=self.args["ep_len"],
                min_timesteps_per_batch=self.args['batch_size'],
                params=params,
                input_shape = self.input_shape
            )
            self.agent_env = MTEnvCollector(
                env=self.env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                args=args,
                params=params,
                example_embedding=example_embedding,
                plot_prefix=self.plot_prefix
            )
            self.mt_flag = True
        
        # MT10 hard
        elif self.params["env_name"] == "mt10_hard":
            self.expert_env = MT10HardCollector(
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                expert_dict=expert_dict,
                device=params['general_setting']['device'],
                max_path_length=self.args["ep_len"],
                min_timesteps_per_batch=self.args['batch_size'],
                params=params,
                input_shape = self.input_shape
            )
            self.agent_env = MTEnvCollector(
                env=self.env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                args=args,
                params=params,
                example_embedding=example_embedding,
                plot_prefix=self.plot_prefix
            )
            self.mt_flag = True

        elif self.params["env_name"] == "mt40":
            self.expert_env = MT40Collector(
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                expert_dict=expert_dict,
                device=params['general_setting']['device'],
                max_path_length=self.args["ep_len"],
                min_timesteps_per_batch=self.args['batch_size'],
                params=params,
                input_shape = self.input_shape
            )
            self.agent_env = MTEnvCollector(
                env=self.env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                args=args,
                params=params,
                example_embedding=example_embedding,
                plot_prefix=self.plot_prefix
            )
            self.mt_flag = True
        
        # MT50
        elif self.params["env_name"] == "mt50":
            self.expert_env = MT50SingleCollector(
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                expert_dict=expert_dict,
                device=params['general_setting']['device'],
                max_path_length=self.args["ep_len"],
                min_timesteps_per_batch=self.args['batch_size'],
                params=params,
                input_shape = self.input_shape
            )
            self.agent_env = MTEnvCollector(
                env=self.env,
                env_cls=env_cls,
                env_args=env_args,
                env_info=self.env_info,
                args=args,
                params=params,
                example_embedding=example_embedding,
                plot_prefix=self.plot_prefix
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
        
        
        if self.mt_flag == False:
            self.plot_prefix += self.args["task_name"] + "_"
    
    
    def run_training_loop(self, n_iter, multiple_samples, baseline=False, expert_task_curve={}, agent_task_curve={}):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """
        
        self.baseline=baseline

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        loss_curve = []
        expert_success_curve = []
        agent_success_curve = []
        self.alternate = 0 # start with base policy training
    
        # TRAIN
        for itr in range(n_iter):
            
            start = time.time()
            train_start_time = time.time()
            
            # collect trajectories, to be used for training
            if itr == 0:
                print("\n\n-------------------------------- Iteration %i -------------------------------- "%itr)
                render = self.params["general_setting"]["train_render"] if (itr % self.args["render_interval"] == 0) else False
                training_returns = self.expert_env.sample_expert(render=False, render_mode="rgb_array", log=True, log_prefix = self.plot_prefix, multiple_samples=multiple_samples)
                # for i in range(5):
                #     training_returns = self.expert_env.sample_expert(render=render, render_mode="rgb_array", log=True, log_prefix = self.plot_prefix)
                # print("sampling expert data for 5 iterations; Ending program")
                # exit(0)
                paths, envsteps_this_batch, infos = training_returns
                self.total_envsteps += envsteps_this_batch
                print("total training samples: ", self.total_envsteps)

                # add collected data to replay buffer
                if self.mt_flag: 
                    self.agent.add_mt_to_replay_buffer(paths)
                else:
                    self.agent.add_to_replay_buffer(paths)
            
            # train agent (using sampled data from replay buffer)
            training_logs = self.train_agent(alternate_flag = self.args["alternate_train"])  # whether or not do alternate training
            self.alternate = 0 if self.alternate==1 else 1
            
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
            

            # EVALUATION
            eval_start_time = time.time()
            if itr % self.args["eval_interval"] == 0:
                print("\n\n-------------------------------- Iteration %i -------------------------------- "%itr)
                render = self.params["general_setting"]["eval_render"]
                if self.mt_flag == False:
                    eval_success_rate = self.expert_env.sample_agent(agent_policy=self.agent.actor, n_sample=self.params["general_setting"]["eval_episodes"], render=render, render_mode="rgb_array", log=True, log_prefix = self.plot_prefix, n_iter=itr)
                    agent_success_curve.append(eval_success_rate)
                else:
                    if self.baseline:
                        eval_infos = self.expert_env.sample_agent(agent_policy=self.agent.actor, n_sample=self.params["general_setting"]["eval_episodes"], render=render, render_mode="rgb_array", log=True, log_prefix = self.plot_prefix, n_iter=itr)
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
                for log in training_logs:
                    print("loss: ", log["Training Loss"])
            
            if min_loss < 0.0001:
                print("\n\n-------------------------------- Training stopped due to early stopping -------------------------------- ")
                print("min loss: ", min_loss)
                break
        
        # TEST
        print("\n\n-------------------------------- Test Results -------------------------------- ")
        render = False
        if self.mt_flag == False:
            eval_success_rate = self.expert_env.sample_agent(agent_policy=self.agent.actor, n_sample=self.params["general_setting"]["eval_episodes"], render=render, render_mode="rgb_array", log=True, log_prefix = self.plot_prefix, n_iter="final")
            print("mean_success_rate: ", eval_success_rate)
        else:
            if self.baseline:
                eval_infos = self.expert_env.sample_agent(agent_policy=self.agent.actor, n_sample=self.params["general_setting"]["eval_episodes"], render=render, render_mode="rgb_array", log=True, log_prefix = self.plot_prefix, n_iter="final")
            else:
                if self.index_flag:
                    eval_infos = self.agent_env.sample_agent(log_prefix=self.plot_prefix, agent_policy=self.agent.actor.policy, input_shape = self.agent.actor.input_shape, render=render)
                else:
                    eval_infos = self.agent_env.sample_agent(log_prefix=self.plot_prefix, agent_policy=self.agent.actor.policy, input_shape = self.agent.actor.input_shape, render=render, plot_weights=True)
            
            for name in eval_infos.keys():
                if name == "mean_success_rate":
                    agent_success_curve.append(eval_infos["mean_success_rate"])
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
   
        return agent_success_curve
   
    def train_agent(self, alternate_flag):
        all_logs = []
 
        for _ in range(self.args['gradient_steps']):

            # sample some data from the data buffer, and train on that batch
            if self.mt_flag:
                if self.index_flag:
                    ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch, index_batch = self.agent.mt_sample(self.args['train_batch_size'])
                    train_log = self.agent.train(ob_batch, ac_batch, index_batch)
                else:  
                    ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch, embedding_batch = self.agent.mt_sample(self.args['train_batch_size'])
                    if alternate_flag:
                        train_log = self.agent.train(ob_batch, ac_batch, embedding_batch, self.alternate)
                    else:
                        train_log = self.agent.train(ob_batch, ac_batch, embedding_batch) # alternate=-1, meaning disabled
            else:
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.args['train_batch_size'])
                train_log = self.agent.train(ob_batch, ac_batch)
            all_logs.append(train_log)
            
        return all_logs
            
    def plot_single_curve(self, curve, tag, plot_prefix, task_name):
        iteration = range(1, len(curve)+1)
        # smooth
        # curve_hat = savgol_filter(curve, 5, 3)
        # data = pd.DataFrame(curve_hat, iteration)
        
        # data = pd.DataFrame(curve, iteration)
        # ax=sns.lineplot(data=data)
        # ax.set_xlabel("Iteration")
        # ax.set_ylabel(tag + " success")
        # ax.set_title(tag + "_" + task_name + " Mean Success Curve")
        # ax.set(ylim=(-0.1, 1.1))
        
        # fig = ax.get_figure()
        # fig.savefig(plot_prefix + task_name + "_"+ tag + "_success_curve.png")
        
        # fig.clf()
        
    def plot_success_curve(self, curve, tag, plot_prefix):
        iteration = range(1, len(curve)+1)
        
        # log success_rate for future usage
        import json
        success_dict = {}
        for i in range(len(curve)):
            success_dict[i] = curve[i]
        success_json = json.dumps(success_dict, sort_keys=False, indent=4)
        f = open(self.plot_prefix + "_success.json", 'w')
        f.write(success_json)
        
        # smooth
        curve_hat = savgol_filter(curve, 3, 2)
        data = pd.DataFrame(curve_hat, iteration)
        ax=sns.lineplot(data=data)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(tag + " success")
        ax.set_title(tag + " Mean Success Curve")
        ax.set(ylim=(-0.1, 1.1))
        
        fig = ax.get_figure()
        fig.savefig(plot_prefix + tag + "_success_curve.png")
        
        fig.clf()
