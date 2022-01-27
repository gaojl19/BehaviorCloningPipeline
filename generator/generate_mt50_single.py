from metaworld.envs.mujoco.env_dict import HARD_MODE_CLS_DICT, HARD_MODE_ARGS_KWARGS

with open("../train_mt50_single.sh", "w") as fout:
    for a in HARD_MODE_CLS_DICT.keys():
        for task_name in HARD_MODE_CLS_DICT[a].keys():
            command = '''python train_single.py \
                --expert_policy_file policy/expert/MT50_Fixed/''' + task_name + '''.pth \
                --exp_name bc_reach \
                --n_iter 100 \
                --eval_interval 200 \
                --learning_rate 1e-4 \
                --video_log_freq -1 \
                --ep_len 200 \
                --batch_size 64 \
                --train_batch_size 32 \
                --config config/BC.json \
                --task_name ''' + task_name + ''' \
                --id MT50_Single_Task \
                --seed 32 \
                --worker_nums 1 \
                --eval_worker_nums 1 \
                --task_env MT50_task_env \
                --no_cuda '''
            fout.write(command + "\n")

fout.close()
        