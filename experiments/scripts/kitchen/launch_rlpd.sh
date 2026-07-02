export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:kitchen_wsrl \
--project baselines-section \
--num_offline_steps 0 \
--offline_data_ratio 0.5 \
--reward_scale 1.0 \
--reward_bias -4.0 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5_000 \
$@
