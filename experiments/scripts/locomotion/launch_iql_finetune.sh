export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python3 finetune.py \
--agent iql \
--exp_name NR5 \
--project NR5 \
--config experiments/configs/train_config.py:locomotion_iql \
--env mujoco/hopper/medium-v0 \
--use_redq \
--seed 0 \
--save_interval 250000 \
--save_dir ./checkpoints/iql \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
$@
