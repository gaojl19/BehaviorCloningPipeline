#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G

#SBATCH -- gres=gpu:1

echo "SLURM_JOBID="$SLURM_JOBID
echo "working directory="$SLURM_SUBMIT_DIR

srun python3 train_mt_baseline.py \
    --expert_policy_file policy/expert/MT10_Medium/ \
	--exp_name bc_reach \
    --n_iter 600000 \
    --eval_interval 1000 \
    --render_interval 40 \
    --learning_rate 1e-4 \
	--video_log_freq -1 \
    --ep_len 200 \
    --batch_size 64 \
    --train_batch_size 32 \
    --config config/soft_module_fixed_mt10_medium.json \
    --id MT50_Single_Task \
    --seed 1920 \
    --worker_nums 1 \
    --eval_worker_nums 1 \
    --task_env MT50_task_env \
    --no_cuda 
echo "Done"