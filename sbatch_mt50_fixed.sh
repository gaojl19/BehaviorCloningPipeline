#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=64G

#SBATCH -- gres=gpu:1

#SBATCH --job-name="MT50-Fixed"

echo "SLURM_JOBID="$SLURM_JOBID
echo "working directory="$SLURM_SUBMIT_DIR

srun python3 train_soft_module.py \
    --expert_policy_file  ../Soft-Module/log/MT50_Fixed_Modular_Deep/mt50/42/model/model_pf_best.pth \
	--exp_name mt50_fixed \
    --n_iter 10 \
    --learning_rate 1e-4 \
	--video_log_freq -1 \
    --ep_len 1 \
    --batch_size 64 \
    --train_batch_size 32 \
    --config config/soft_module_fixed_mt50.json \
    --id MT50_Fixed \
    --seed 32 \
    --worker_nums 1 \
    --eval_worker_nums 1 \
    --task_env MT50_task_env \
echo "Done"