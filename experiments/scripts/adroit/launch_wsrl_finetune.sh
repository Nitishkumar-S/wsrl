export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

# env: pen-binary-v0, door-binary-v0, relocate-binary-v0

python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:adroit_wsrl \
--project method-section \
--num_offline_steps 20_000 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--env pen-binary-v0 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
$@
