export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python3 finetune.py \
--agent iql \
--config experiments/configs/train_config.py:kitchen_iql \
--project baselines-section \
--group no-redq-utd1 \
--num_offline_steps 250_000 \
--reward_scale 1.0 \
--reward_bias -4.0 \
--env kitchen-mixed-v0 \
$@
