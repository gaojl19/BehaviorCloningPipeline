import os
import time
import torch
import numpy as np
import gym

from torch_rl.rl_trainer import RL_Trainer
from agents.bc_agent import MLPAgent
from policy.loaded_gaussian_policy import LoadedGaussianPolicy
from utils.args import get_params
from utils.logger import Logger
from networks.base import MLPBase


from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT, EASY_MODE_ARGS_KWARGS
from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, HARD_MODE_ARGS_KWARGS
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
        self.args['agent_class'] = MLPAgent ## HW1: you will modify this
        self.args['agent_params'] = agent_params
        
        # BUILD ENV
        self.device = torch.device("cuda:{}".format(args["device"]) if args["cuda"] else "cpu")
        
        if args["task_env"] == "MT10_task_env":
            cls_dicts = {args["task_name"]: EASY_MODE_CLS_DICT[args["task_name"]]}
            cls_args = {args["task_name"]: EASY_MODE_ARGS_KWARGS[args["task_name"]]}
            env_name =  EASY_MODE_CLS_DICT[args["task_name"]]
            
            # overwrite if random_init=True, else use default setting
            if args["random_init"] == True:
                cls_args[args["task_name"]]['kwargs']['random_init']=True
            print(cls_args)
            

        elif args["task_env"] == "MT50_task_env":
            if args["task_name"] in HARD_MODE_CLS_DICT['train'].keys():     # 45 tasks
                cls_dicts = {args["task_name"]: HARD_MODE_CLS_DICT['train'][args["task_name"]]}
                cls_args = {args["task_name"]: HARD_MODE_ARGS_KWARGS['train'][args["task_name"]]}
                env_name =  HARD_MODE_CLS_DICT['train'][args["task_name"]]
            else:
                cls_dicts = {args["task_name"]: HARD_MODE_CLS_DICT['test'][args["task_name"]]}
                cls_args = {args["task_name"]: HARD_MODE_ARGS_KWARGS['test'][args["task_name"]]}
                env_name =  HARD_MODE_CLS_DICT['test'][args["task_name"]]

        else:
            raise NotImplementedError
    
    
        # set cls_args['kwargs']['obs_type'] = params['meta_env']['obs_type']
        # also here you could set random init
        cls_args[args["task_name"]]['kwargs']['obs_type'] = params['meta_env']['obs_type']
        cls_args[args["task_name"]]['kwargs']['random_init'] = params['meta_env']['random_init']
    
        # env, cls_dicts, cls_args = get_meta_env(params['env_name'], params['env'], params['meta_env'])
        self.env = get_meta_env(env_name, params['env'], params['meta_env'], return_dicts=False) 

        self.env.seed(args["seed"])
        torch.manual_seed(args["seed"])
        np.random.seed(args["seed"])
        if args["cuda"]:
            torch.backends.cudnn.deterministic=True

        self.experiment_name = os.path.split( os.path.splitext( args["config"] )[0] )[-1] if args["id"] is None \
            else args["id"]
        self.logger = Logger(args['log_dir'])

        params['general_setting']['env'] = self.env
        params['general_setting']['logger'] = self.logger
        params['general_setting']['device'] = self.device
        params['net']['base_type']=MLPBase
        
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.args['agent_params']['discrete'] = discrete
        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.args['agent_params']['ac_dim'] = ac_dim
        self.args['agent_params']['ob_dim'] = ob_dim
        
        # self.args['ep_len'] = self.args['ep_len']
        print(self.args)
        print(params)
        self.params = params
        
        # LOAD EXPERT POLICY
        print('Loading expert policy from...', self.args['expert_policy_file'])
        self.loaded_expert_policy = LoadedGaussianPolicy(env=self.env, params=params, policy_path=args["expert_policy_file"])
        expert_dict = {self.args["task_name"]: self.loaded_expert_policy}
        print('Done restoring expert policy...')

    
        # RL TRAINER
        self.rl_trainer = RL_Trainer(env = self.env, env_cls = cls_dicts, env_args = [params["env"], cls_args, params["meta_env"]], args = self.args, params = params, expert_dict=expert_dict) ## HW1: you will modify this
        
        
    
    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            n_iter=self.args['n_iter'],
            baseline=False
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
    print("no cuda: ", args.no_cuda)
    print("gpu available: ", torch.cuda.is_available())
    if not args.cuda:
        args.device = "cpu"
    # if not args.cuda:
    #     args.device = "cpu"
    params = get_params(args.config)
    
    # CREATE DIRECTORY FOR LOGGING
    if args.do_dagger:
        logdir_prefix = 'DAgger_'
        print("do dagger")
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')
    else:
        logdir_prefix = 'Base_'
        # assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')

    
    # directory for logging
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)
    logdir = logdir_prefix + args.exp_name + '_' + args.task_name + '_' + str(args.n_layers) + str(args.batch_size) + str(args.learning_rate) + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)
    
     # convert args to dictionary
    args = vars(args)
    
    args['log_dir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)


    # RUN TRAINING
    trainer = BC_Trainer(args, params)
    trainer.run_training_loop()

if __name__ == "__main__":
    main()