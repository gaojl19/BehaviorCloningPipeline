#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G

#SBATCH -- gres=gpu:1

#SBATCH --job-name="MT10-reach-v1"
source batch_scripts/MT10_Fixed/seed.sh
echo "SLURM_JOBID="$SLURM_JOBID
echo "working directory="$SLURM_SUBMIT_DIR
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
xvfb-run -a -s "-screen 0 1400x900x24" bash

srun python3 train.py     --expert_policy_file ../Multi-Task-RL/log/MT10_Single_Task/reach-v1/Fixed/257/model/model_pf_best.pth 	--exp_name bc_reach --n_iter 40     --learning_rate 1e-4 	--video_log_freq -1     --ep_len 200     --batch_size 64     --train_batch_size 32     --config config/BC.json     --task_name reach-v1     --id MT10_Single_Task     --seed 257     --worker_nums 1     --eval_worker_nums 1     --task_env MT10_task_env 
    
echo "Done"