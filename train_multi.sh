python train_soft_module.py \
    --expert_policy_file policy/expert/MT10_Similar/ \
	--exp_name bc_reach \
    --n_iter 2000 \
    --eval_interval 1000 \
    --render_interval 40 \
    --learning_rate 1e-4 \
	--video_log_freq -1 \
    --ep_len 200 \
    --batch_size 64 \
    --train_batch_size 32 \
    --config config/soft_module_fixed_mt10_similar.json \
    --id MT50_Single_Task \
    --seed 32 \
    --worker_nums 1 \
    --eval_worker_nums 1 \
    --task_env MT50_task_env \
    --no_cuda \
    --multiple_runs True

# python train_mh_sac.py \
#     --expert_policy_file policy/expert/MT10_Similar/ \
# 	--exp_name bc_reach \
#     --n_iter 2000 \
#     --eval_interval 1000 \
#     --render_interval 40 \
#     --learning_rate 1e-4 \
# 	--video_log_freq -1 \
#     --ep_len 200 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/mtmh_sac_fixed_mt10_similar.json \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --no_cuda \
#     --multiple_runs True


# python train_mt_baseline.py \
#     --expert_policy_file policy/expert/MT10_Similar/ \
# 	--exp_name bc_reach \
#     --n_iter 2000 \
#     --eval_interval 1000 \
#     --render_interval 40 \
#     --learning_rate 1e-4 \
# 	--video_log_freq -1 \
#     --ep_len 200 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/soft_module_fixed_mt10_similar.json \
#     --id MT50_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT50_task_env \
#     --no_cuda \
#     --multiple_runs True