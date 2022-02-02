#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G

#SBATCH -- gres=gpu:1

#SBATCH --job-name="MT50"
echo "SLURM_JOBID="$SLURM_JOBID
echo "working directory="$SLURM_SUBMIT_DIR

srun python3 train_single.py     --expert_policy_file ../Multi-Task-RL/log/MT50_Single_Task/lever-pull-v1/Fixed/238/model/model_pf_best.pth 	--exp_name bc_reach --n_iter 600000     --eval_interval 1000     --learning_rate 1e-4 	--video_log_freq -1     --ep_len 200     --batch_size 64     --train_batch_size 32     --config config/BC.json     --task_name lever-pull-v1     --id MT50_Single_Task     --seed 32     --worker_nums 1     --eval_worker_nums 1     --task_env MT50_task_env
    
echo "Done"