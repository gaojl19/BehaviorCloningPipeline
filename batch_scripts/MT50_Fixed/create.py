import os
import json

if __name__ == "__main__":
    with open("../../metaworld_utils/MT50_task_env.json", "r") as f:
        task_env = json.load(f)

    print(task_env)

    for task in task_env.keys():
        batch = '''#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G

#SBATCH -- gres=gpu:1

#SBATCH --job-name="MT50"
echo "SLURM_JOBID="$SLURM_JOBID
echo "working directory="$SLURM_SUBMIT_DIR

srun python3 train_single.py \
    --expert_policy_file ../Multi-Task-RL/log/MT50_Single_Task/''' + task + '''/Fixed/238/model/model_pf_best.pth \
	--exp_name bc_reach --n_iter 600000 \
    --eval_interval 1000 \
    --learning_rate 1e-4 \
	--video_log_freq -1 \
    --ep_len 200 \
    --batch_size 64 \
    --train_batch_size 32 \
    --config config/BC.json \
    --task_name ''' + task + ''' \
    --id MT50_Single_Task \
    --seed 32 \
    --worker_nums 1 \
    --eval_worker_nums 1 \
    --task_env MT50_task_env
    
echo "Done"'''

        with open(task.replace("-","_") +".sh", "w") as fout:
            fout.write(batch)
        fout.close()

    with open("../../MT50_submit.sh", "w") as fout:
        for task in task_env.keys():
            string = "sbatch batch_scripts/MT50_Fixed/"+task.replace("-","_")+".sh"
            fout.write(string+"\n")
