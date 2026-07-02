export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python3 finetune.py \
--agent iql \
--config experiments/configs/train_config.py:locomotion_iql \
--env halfcheetah-medium-replay-v2 \
--project locomotion-finetune \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
$@
