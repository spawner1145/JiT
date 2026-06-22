torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --accum_iter 8 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir jit_b16_bs1024 \
--resume jit_b16_bs1024 \
--wandb --wandb_project JiT --wandb_run_name jit_b16_bs1024 \
--data_path /root/autodl-tmp/study/imgnet_data \
--online_eval