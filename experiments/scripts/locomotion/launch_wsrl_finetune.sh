export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:locomotion_wsrl \
--project method-section \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 250_000 \
--env halfcheetah-medium-replay-v2 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
$@
