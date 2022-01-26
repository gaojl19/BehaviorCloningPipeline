# python train_soft_module.py \
#     --expert_policy_file policy/expert/MT10/ \
# 	--exp_name bc_reach \
#     --n_iter 40 \
#     --render_interval 40 \
#     --learning_rate 1e-4 \
# 	--video_log_freq -1 \
#     --ep_len 200 \
#     --batch_size 64 \
#     --train_batch_size 32 \
#     --config config/soft_module_fixed_mt10.json \
#     --id MT10_Single_Task \
#     --seed 32 \
#     --worker_nums 1 \
#     --eval_worker_nums 1 \
#     --task_env MT10_task_env \
#     --no_cuda 

python train_mt_baseline.py \
    --expert_policy_file policy/expert/MT50_Fixed/ \
	--exp_name bc_reach \
    --n_iter 1000 \
    --eval_interval 500 \
    --render_interval 40 \
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
    --no_cuda \