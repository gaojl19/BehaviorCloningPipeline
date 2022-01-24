import json

if __name__ == "__main__":
    with open("../metaworld_utils/MT50_task_env.json", "r") as f:
        task_env = json.load(f)

    print(task_env)

    # LOAD MT50 FIXED
    with open("load_mt50_fixed_model.sh", "w") as fout:
        for task in task_env.keys():
            command = "cp ../../Multi-Task-RL/log/MT50_Single_Task/" + task + "/Fixed/238/model/model_pf_best.pth ../policy/expert/MT50_Fixed/" + task + ".pth"
            fout.write(command + "\n")
        fout.close()
        
    # LOAD MT50 RANDOM
    with open("load_mt50_random_model.sh", "w") as fout:
        for task in task_env.keys():
            command = "cp ../../Multi-Task-RL/log/MT50_Single_Task/" + task + "/Random/238/model/model_pf_best.pth ../policy/expert/MT50_Random/" + task + ".pth"
            fout.write(command + "\n")
        fout.close()
    
    # sbatch submit MT50 Fixed
    with open("../MT50_Fixed_submit.sh", "w") as fout:
        batch = '''#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G

#SBATCH -- gres=gpu:1

echo "SLURM_JOBID="$SLURM_JOBID
echo "working directory="$SLURM_SUBMIT_DIR

srun python3 train_soft_module.py \
        --expert_policy_file policy/expert/MT50_Fixed/ \
	    --exp_name mt50_fixed \
        --n_iter 500 \
        --render_interval 500 \
        --learning_rate 1e-4 \
	    --video_log_freq -1 \
        --ep_len 200 \
        --batch_size 64 \
        --train_batch_size 32 \
        --config config/soft_module_fixed_mt50.json \
        --id MT50_Single_Task \
        --seed 32 \
        --worker_nums 1 \
        --eval_worker_nums 1 \
        --task_env MT50_task_env \
        --no_cuda
    
echo "Done" '''
        
        fout.write(batch)
        fout.close()
    
    
    # sbatch submit MT50 Random
    with open("../MT50_Random_submit.sh", "w") as fout:
        batch = '''#!/bin/bash
#SBATCH --partition=viscam --qos=normal
#SBATCH --nodes=1
#SBATCH --mem=16G

#SBATCH -- gres=gpu:1

echo "SLURM_JOBID="$SLURM_JOBID
echo "working directory="$SLURM_SUBMIT_DIR

srun python3 train_soft_module.py \
        --expert_policy_file policy/expert/MT50_Random/ \
	    --exp_name mt50_random \
        --n_iter 500 \
        --render_interval 500 \
        --learning_rate 1e-4 \
	    --video_log_freq -1 \
        --ep_len 200 \
        --batch_size 64 \
        --train_batch_size 32 \
        --config config/soft_module_random_mt50.json \
        --id MT50_Single_Task \
        --seed 32 \
        --worker_nums 1 \
        --eval_worker_nums 1 \
        --task_env MT50_task_env \
        --no_cuda
    
echo "Done" '''
        
        fout.write(batch)
        fout.close()