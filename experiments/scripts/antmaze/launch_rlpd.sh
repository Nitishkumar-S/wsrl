export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python3 finetune.py \
--agent sac \
--config experiments/configs/train_config.py:antmaze_wsrl \
--project baselines-section \
--config.agent_kwargs.critic_subsample_size 1 \
--reward_scale 10.0 \
--reward_bias -5.0 \
--num_offline_steps 0 \
--num_online_steps 500_000 \
--offline_data_ratio 0.5 \
--env antmaze-large-diverse-v2 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
$@
