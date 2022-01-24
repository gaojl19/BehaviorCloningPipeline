#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G

#SBATCH -- gres=gpu:1

#SBATCH --job-name="MT50"
source batch_scripts/MT50_batch/seed.sh
echo "SLURM_JOBID="$SLURM_JOBID
echo "working directory="$SLURM_SUBMIT_DIR

srun python3 train.py     --expert_policy_file ../Multi-Task-RL/log/MT50_Single_Task/faucet-open-v1/Random/238/model/model_pf_best.pth 	--exp_name bc_reach --n_iter 50     --learning_rate 1e-4 	--video_log_freq -1     --ep_len 200     --batch_size 64     --train_batch_size 32     --config config/BC_random.json     --task_name faucet-open-v1     --id MT50_Single_Task     --seed 32     --worker_nums 1     --eval_worker_nums 1     --task_env MT50_task_env     --no_cuda 
    
echo "Done"