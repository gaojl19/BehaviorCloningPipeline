python train_soft_module.py \
    --expert_policy_file policy/expert/MT10_Hard/ \
	--exp_name bc_reach \
    --n_iter 0 \
    --load_from_checkpoint fig/mt10_hard_2_2_256_dpModule_hybrid_fixed/model.pth \
    --eval_interval 1 \
    --render_interval 40 \
    --learning_rate 1e-4 \
	--video_log_freq -1 \
    --ep_len 200 \
    --batch_size 64 \
    --train_batch_size 32 \
    --config config/soft_module_fixed_mt10_hard.json \
    --id MT50_Single_Task \
    --seed 32 \
    --worker_nums 1 \
    --eval_worker_nums 1 \
    --task_env MT50_task_env \
    --no_cuda 

# python3 train_soft_module.py         --expert_policy_file policy/expert/MT50_Fixed/ 	    --exp_name mt50_fixed         --n_iter 500         --render_interval 500         --learning_rate 1e-4 	    --video_log_freq -1         --ep_len 200         --batch_size 64         --train_batch_size 32         --config config/soft_module_fixed_mt50.json         --id MT50_Single_Task         --seed 32         --worker_nums 1         --eval_worker_nums 1         --task_env MT50_task_env         --no_cuda
    