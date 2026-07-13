export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python3 finetune.py \
--agent sac \
--exp_name NR5 \
--config experiments/configs/train_config.py:locomotion_wsrl \
--project NR5 \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
--env mujoco/hopper/medium-v0 \
--seed 0 \
--save_interval 250000 \
--utd 4 \
--batch_size 256 \
--warmup_steps 5000 \
--save_dir ./checkpoints/wsrl \
$@
