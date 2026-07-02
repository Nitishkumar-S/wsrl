export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

# env: pen-binary-v0, door-binary-v0, relocate-binary-v0

python finetune.py \
--agent cql \
--config experiments/configs/train_config.py:adroit_cql \
--project baselines-section \
--group no-redq-utd1 \
--warmup_steps 0 \
--num_offline_steps 20_000 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--env pen-binary-v0 \
$@
