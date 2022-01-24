#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G

#SBATCH -- gres=gpu:1

echo "SLURM_JOBID="$SLURM_JOBID
echo "working directory="$SLURM_SUBMIT_DIR

srun python3 train_soft_module.py         --expert_policy_file policy/expert/MT50_Fixed/ 	    --exp_name mt50_fixed         --n_iter 500         --render_interval 500         --learning_rate 1e-4 	    --video_log_freq -1         --ep_len 200         --batch_size 64         --train_batch_size 32         --config config/soft_module_fixed_mt50.json         --id MT50_Single_Task         --seed 32         --worker_nums 1         --eval_worker_nums 1         --task_env MT50_task_env         --no_cuda
    
echo "Done" 