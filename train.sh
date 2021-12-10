# python train.py \
#     --expert_policy_file policy/expert/sweep-into-v1.pth \
# 	--exp_name bc_reach --n_iter 10 \
#     --learning_rate 1e-4 \
# 	--video_log_freq -1 \
#     --ep_len 200 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/BC.json \
#     --task_name sweep-into-v1 \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --no_cuda \

python train.py \
    --expert_policy_file policy/expert/reach_v1_pf.pth \
	--exp_name bc_reach --n_iter 10 \
    --learning_rate 1e-4 \
	--video_log_freq -1 \
    --ep_len 200 \
    --batch_size 64 \
    --train_batch_size 32 \
    --config config/BC.json \
    --task_name reach-v1 \
    --id MT10_Single_Task \
    --seed 32 \
    --worker_nums 1 \
    --eval_worker_nums 1 \
    --task_env MT10_task_env \
    --do_dagger
    --no_cuda \