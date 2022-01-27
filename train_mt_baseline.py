import os
import time
import torch
import numpy as np
import gym

from torch_rl.rl_trainer import RL_Trainer
from agents.bc_agent import MLPAgent, SoftModuleAgent
from policy.loaded_gaussian_policy import LoadedGaussianPolicy
from utils.args import get_params
from utils.logger import Logger
from networks.base import MLPBase
import random


from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT
from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT
from metaworld_utils.customize_env_dict import DIVERSE_MT10_CLS_DICT, SIMILAR_MT10_CLS_DICT, FAIL_MT10_CLS_DICT
from metaworld_utils.meta_env import get_meta_env

class BC_Trainer(object):

    def __init__(self, args, params):

        # Agent params
        agent_params = {
            'n_layers': args['n_layers'],
            'size': args['size'],
            'learning_rate': args['learning_rate'],
            'max_replay_buffer_size': args['max_replay_buffer_size'],
            }

        self.args = args
        
        # for baseline
        self.args['agent_class'] = MLPAgent
        self.args['agent_params'] = agent_params
        
        # BUILD ENV
        self.device = torch.device("cuda:{}".format(args["device"]) if args["cuda"] else "cpu")
        env, cls_dicts, cls_args = get_meta_env( params['env_name'], params['env'], params['meta_env'])
        
        self.env = env
        self.env.seed(args['seed'])
        torch.manual_seed(args['seed'])
        np.random.seed(args['seed'])
        random.seed(args['seed'])
        if args['cuda']:
            torch.backends.cudnn.deterministic=True
    

        params['general_setting']['env'] = self.env
        params['general_setting']['device'] = self.device

        params['net']['base_type'] = MLPBase

        import torch.multiprocessing as mp
        mp.set_start_method('spawn', force=True)

        example_embedding = self.env.active_task_one_hot 
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.args['agent_params']['discrete'] = discrete
        
        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.args['agent_params']['ac_dim'] = ac_dim
        self.args['agent_params']['ob_dim'] = ob_dim
        
        device = torch.device("cuda:{}".format(args['device']) if args['cuda'] else "cpu")

        # self.args['ep_len'] = self.args['ep_len']
        print(self.args)
        print(params)
        self.params = params
        
        self.agent_task_curve={}
        self.expert_task_curve={}
        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        
        # LOAD EXPERT POLICY
        print('Loading expert policy from...', self.args['expert_policy_file'])
        expert_dict = {}
        if self.params["env_name"] == "mt10":
            for name, env_name in EASY_MODE_CLS_DICT.items():
                # add necessary env args
                expert_env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
                expert_env.seed(args["seed"])
                params['expert_net']['base_type']=MLPBase
                
                file_path = self.args['expert_policy_file'] + name + ".pth"
                if os.path.exists(file_path):
                    expert_dict[name] = LoadedGaussianPolicy(env=expert_env, params=params, policy_path=file_path)
                
                self.expert_task_curve[name + "_success_rate"] = []
                self.agent_task_curve[name + "_success_rate"] = []
                
        elif self.params["env_name"] == "mt50":
            # load MT50 train environment
            for name, env_name in HARD_MODE_CLS_DICT["train"].items():
                expert_env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
                expert_env.seed(args["seed"])
                params['expert_net']['base_type']=MLPBase
                
                
                file_path = self.args['expert_policy_file'] + name + ".pth"
                if os.path.exists(file_path):
                    expert_dict[name] = LoadedGaussianPolicy(env=expert_env, params=params, policy_path=file_path)
                
                self.expert_task_curve[name + "_success_rate"] = []
                self.agent_task_curve[name + "_success_rate"] = []
            
            # load MT50 test environment
            for name, env_name in HARD_MODE_CLS_DICT["test"].items():
                expert_env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
                expert_env.seed(args["seed"])
                params['expert_net']['base_type']=MLPBase
                
                file_path = self.args['expert_policy_file'] + name + ".pth"
                if os.path.exists(file_path):
                    expert_dict[name] = LoadedGaussianPolicy(env=expert_env, params=params, policy_path=file_path)
                self.expert_task_curve[name + "_success_rate"] = []
                self.agent_task_curve[name + "_success_rate"] = []
        
        
        elif self.params["env_name"] == "mt10_diverse":
            for name, env_name in DIVERSE_MT10_CLS_DICT.items():
                # add necessary env args
                expert_env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
                expert_env.seed(args["seed"])
                params['expert_net']['base_type']=MLPBase
                
                file_path = self.args['expert_policy_file'] + name + ".pth"
                if os.path.exists(file_path):
                    expert_dict[name] = LoadedGaussianPolicy(env=expert_env, params=params, policy_path=file_path)
                    ob_dim = max(ob_dim, expert_dict[name].ob_dim)
                self.expert_task_curve[name + "_success_rate"] = []
                self.agent_task_curve[name + "_success_rate"] = []
            
            
        elif self.params["env_name"] == "mt10_similar":
            for name, env_name in SIMILAR_MT10_CLS_DICT.items():
                # add necessary env args
                expert_env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
                expert_env.seed(args["seed"])
                params['expert_net']['base_type']=MLPBase
                
                file_path = self.args['expert_policy_file'] + name + ".pth"
                if os.path.exists(file_path):
                    expert_dict[name] = LoadedGaussianPolicy(env=expert_env, params=params, policy_path=file_path)
                    ob_dim = max(ob_dim, expert_dict[name].ob_dim)
                self.expert_task_curve[name + "_success_rate"] = []
                self.agent_task_curve[name + "_success_rate"] = []
        
        elif self.params["env_name"] == "mt10_fail":
            for name, env_name in FAIL_MT10_CLS_DICT.items():
                # add necessary env args
                expert_env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 
                expert_env.seed(args["seed"])
                params['expert_net']['base_type']=MLPBase
                
                file_path = self.args['expert_policy_file'] + name + ".pth"
                if os.path.exists(file_path):
                    expert_dict[name] = LoadedGaussianPolicy(env=expert_env, params=params, policy_path=file_path)
                    ob_dim = max(ob_dim, expert_dict[name].ob_dim)
                self.expert_task_curve[name + "_success_rate"] = []
                self.agent_task_curve[name + "_success_rate"] = []
        
        
        print('Done restoring expert policy...')
        self.args['agent_params']['ac_dim'] = ac_dim
        self.args['agent_params']['ob_dim'] = ob_dim
        
        # RL TRAINER
        self.rl_trainer = RL_Trainer(
            env = self.env,
            env_cls = cls_dicts, 
            env_args = [params["env"], cls_args, params["meta_env"]], 
            args = self.args, 
            params = params, 
            expert_dict = expert_dict, 
            input_shape = ob_dim,
            example_embedding=example_embedding
        )


    
    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            n_iter=self.args['n_iter'],
            baseline=True,
            expert_task_curve=self.expert_task_curve,
            agent_task_curve=self.agent_task_curve
        )


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # relative to where you're running this script from
    # parser.add_argument('--expert_data', '-ed', type=str, required=True) #relative to where you're running this script from
    # parser.add_argument('--env_name', '-env', type=str, help='choices: Ant-v2, Humanoid-v2, Walker-v2, HalfCheetah-v2, Hopper-v2', required=True)
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)
    parser.add_argument('--do_dagger', action='store_true')
    parser.add_argument('--ep_len', type=int)

    parser.add_argument('--gradient_steps', type=int, default=1)  # number of gradient steps for training policy (per iter in n_iter)
    parser.add_argument('--n_iter', '-n', type=int, default=1)
    parser.add_argument('--render_interval', type=int, default=1)
    parser.add_argument('--eval_interval', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=64)  # training data collected (in the env) during each iteration
    parser.add_argument('--eval_batch_size', type=int,
                        default=32)  # eval data collected (in the env) for logging metrics
    parser.add_argument('--train_batch_size', type=int,
                        default=32)  # number of sampled data points to be used per gradient/train step

    parser.add_argument('--n_layers', type=int, default=2)  # depth, of policy to be learned
    parser.add_argument('--size', type=int, default=400)  # width of each layer, of policy to be learned
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)  # LR for supervised learning

    parser.add_argument('--video_log_freq', type=int, default=5)
    parser.add_argument('--scalar_log_freq', type=int, default=1)
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')
    parser.add_argument('--which_gpu', type=int, default=0)
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)
    parser.add_argument('--save_params', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    
    
    parser.add_argument('--worker_nums', type=int, default=4, help='worker nums')
    parser.add_argument('--eval_worker_nums', type=int, default=2,help='eval worker nums')
    parser.add_argument("--config", type=str,   default=None, help="config file", )
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument("--random_init", type=bool, default=False, help="whether use random init when collecting data & evaluating", )
    parser.add_argument("--device", type=int, default=0, help="gpu secification", )

    # tensorboard
    parser.add_argument("--id", type=str,   default=None, help="id for tensorboard", )
    
    # single task learning name
    parser.add_argument("--task_name", type=str, default=None,help="task name for single task training",)
    # single task-env mapping name
    parser.add_argument("--task_env", type=str, default=None, help="task to env mapping for single task training: MT10_task_env / MT50_task_env",)
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if not args.cuda:
        args.device = "cpu"
    params = get_params(args.config)
    
     # convert args to dictionary
    args = vars(args)


    # RUN TRAINING
    trainer = BC_Trainer(args, params)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()