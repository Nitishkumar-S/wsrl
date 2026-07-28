#!/bin/bash
#SBATCH -p ml_gpu-rtx2080
#SBATCH --gres=gpu:1
#SBATCH --nodelist=mlgpu15
# #SBATCH --mem 4000 # memory pool for each core (4GB)
# #SBATCH --mem-per-cpu 4000 # memory pool for each core (4GB)
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -o log/%x.%N.%j.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e log/%x.%N.%j.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J debug # sets the job name. If not specified, the file name will be used as job name

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";
echo "Running job $SLURM_JOB_NAME using $SLURM_JOB_CPUS_PER_NODE cpus per node with given JID $SLURM_JOB_ID on queue $SLURM_JOB_PARTITION";

# Job to perform
cd  /work/dlc2workfs3/nayakk-kartik-lab/wsrl

# Set up venv (if not already created)
# Create the venv if it doesn't exist
if [ ! -d ".venv1" ]; then
  python3 -m venv .venv1
  source .venv1/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  pip uninstall -y jax jaxlib nvidia-cudnn-cu12 nvidia-cublas-cu12
  pip install "jax[cuda12_pip]==0.4.23" "nvidia-cudnn-cu12<9.0" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
else
  source .venv1/bin/activate
fi

python -c "import jax; print(jax.default_backend()); print(jax.devices())"

# Run your pipeline script
# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --env mujoco/walker2d/medium-v0 \
#   --exp_name NR5 \
#   --project NR5 \
#   --use_redq \
#   --seed 0 \
#   --warmup_steps 0 \
#   --num_offline_steps 50 \
#   --num_online_steps 20 \
#   --log_interval 5 \
#   --eval_interval 10 \
#   --save_interval 10 \
#   --n_eval_trajs 5 \
#   --use_redq

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#     --exp_name NR5 \
#     --env mujoco/hopper/medium-v0 \
#     --project NR5 \
#     --seed 2 \
#     --save_interval 250000 \
#     --save_dir ./checkpoints/wsrl \
#     --batch_size 256 \
#     --utd 4

# bash experiments/scripts/locomotion/launch_iql_finetune.sh \
#     --exp_name NR5 \
#     --env mujoco/hopper/medium-v0 \
#     --project NR5 \
#     --seed 0 \
#     --save_interval 250000 \
#     --save_dir ./checkpoints/iql \
#     --use_redq 

# bash experiments/scripts/locomotion/launch_cql_finetune.sh \
#     --exp_name NR5 \
#     --env mujoco/hopper/medium-v0 \
#     --project NR5 \
#     --seed 2 \
#     --save_interval 250000 \
#     --save_dir ./checkpoints/cql \
#     --use_redq \
#     --utd 4


# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#     --exp_name NR5-wsrl-finetune-hopper-seed-0 \
#     --env mujoco/hopper/medium-v0 \
#     --agent sac \
#     --seed 0 \
#     --num_offline_steps 250000 \
#     --num_online_steps 500000 \
#     --resume_path ./checkpoints/wsrl/NR5/NR5-wsrl-cql-pretrain-hopper-seed-0_mujoco-hopper-medium-v0_cql_seed0_20260726_182037_rnd671912

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#     --exp_name NR5-wsrl-finetune-hopper-seed-1 \
#     --env mujoco/hopper/medium-v0 \
#     --agent sac \
#     --seed 1 \
#     --num_offline_steps 250000 \
#     --num_online_steps 500000 \
#     --resume_path ./checkpoints/wsrl/NR5/NR5-wsrl-cql-pretrain-hopper-seed-1_mujoco-hopper-medium-v0_cql_seed1_20260726_190954_rnd230691

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#     --exp_name NR5-wsrl-finetune-hopper-seed-2 \
#     --env mujoco/hopper/medium-v0 \
#     --agent sac \
#     --seed 2 \
#     --num_offline_steps 250000 \
#     --num_online_steps 500000 \
#     --resume_path ./checkpoints/wsrl/NR5/NR5-wsrl-cql-pretrain-hopper-seed-2_mujoco-hopper-medium-v0_cql_seed2_20260726_195914_rnd332012


# bash experiments/scripts/locomotion/launch_iql_finetune.sh \
#   --exp_name NR5-humanoid-iql-seed-0 \
#   --project NR5 \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 0 \
#   --use_redq \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --save_dir ./checkpoints/humanoid/iql


# bash experiments/scripts/locomotion/launch_iql_finetune.sh \
#   --exp_name NR5-humanoid-iql-seed-1 \
#   --project NR5 \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 1 \
#   --use_redq \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --save_dir ./checkpoints/humanoid/iql

# bash experiments/scripts/locomotion/launch_iql_finetune.sh \
#   --exp_name NR5-humanoid-iql-seed-2 \
#   --project NR5 \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 2 \
#   --use_redq \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --save_dir ./checkpoints/humanoid/iql


# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-pretrain-seed-0 \
#   --project NR5 \
#   --agent cql \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 0 \
#   --num_offline_steps 250000 \
#   --num_online_steps 0 \
#   --save_interval 250000 \
#   --save_dir ./checkpoints/humanoid/wsrl

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-pretrain-seed-1 \
#   --project NR5 \
#   --agent cql \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 1 \
#   --num_offline_steps 250000 \
#   --num_online_steps 0 \
#   --save_interval 250000 \
#   --save_dir ./checkpoints/humanoid/wsrl

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-pretrain-seed-2 \
#   --project NR5 \
#   --agent cql \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 2 \
#   --num_offline_steps 250000 \
#   --num_online_steps 0 \
#   --save_interval 250000 \
#   --save_dir ./checkpoints/humanoid/wsrl


# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-finetune-seed-0 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 0 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-0_mujoco-humanoid-medium-v0_cql_seed0_20260727_212403_rnd656513

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-finetune-seed-1 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 1 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-1_mujoco-humanoid-medium-v0_cql_seed1_20260727_212415_rnd823888

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-finetune-seed-2 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 2 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-2_mujoco-humanoid-medium-v0_cql_seed2_20260727_212427_rnd885403


# Ablations Different Warmup Steps

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-ablation-1kWarmup-finetune-seed-0 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 0 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --warmup_steps 1000 \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-0_mujoco-humanoid-medium-v0_cql_seed0_20260727_212403_rnd656513

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-ablation-1kWarmup-finetune-seed-1 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 1 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --warmup_steps 1000 \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-1_mujoco-humanoid-medium-v0_cql_seed1_20260727_212415_rnd823888

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-ablation-1kWarmup-finetune-seed-2 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 2 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --warmup_steps 1000 \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-2_mujoco-humanoid-medium-v0_cql_seed2_20260727_212427_rnd885403


# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-ablation-20kWarmup-finetune-seed-0 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 0 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --warmup_steps 20000 \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-0_mujoco-humanoid-medium-v0_cql_seed0_20260727_212403_rnd656513

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-ablation-20kWarmup-finetune-seed-1 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 1 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --warmup_steps 20000 \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-1_mujoco-humanoid-medium-v0_cql_seed1_20260727_212415_rnd823888

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-ablation-20kWarmup-finetune-seed-2 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 2 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --warmup_steps 20000 \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-2_mujoco-humanoid-medium-v0_cql_seed2_20260727_212427_rnd885403



# Ablations Random Warmup

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-ablation-RandomWarmup-finetune-seed-0 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 0 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --random_warmup True \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-0_mujoco-humanoid-medium-v0_cql_seed0_20260727_212403_rnd656513

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-ablation-RandomWarmup-finetune-seed-1 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 1 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --random_warmup True \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-1_mujoco-humanoid-medium-v0_cql_seed1_20260727_212415_rnd823888

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#   --exp_name NR5-humanoid-wsrl-ablation-RandomWarmup-finetune-seed-2 \
#   --project NR5 \
#   --agent sac \
#   --env mujoco/humanoid/medium-v0 \
#   --seed 2 \
#   --num_offline_steps 250000 \
#   --num_online_steps 500000 \
#   --save_interval 250000 \
#   --random_warmup True \
#   --save_dir ./checkpoints/humanoid/wsrl \
#   --resume_path ./checkpoints/humanoid/wsrl/NR5/NR5-humanoid-wsrl-pretrain-seed-2_mujoco-humanoid-medium-v0_cql_seed2_20260727_212427_rnd885403

# Ablation Different Pretrain Agents

#IQL Pretraining
# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#     --exp_name NR5-humanoid-wsrl-ablation-iql-pretraining-seed-0 \
#     --env mujoco/humanoid/medium-v0 \
#     --agent iql \
#     --seed 0 \
#     --num_offline_steps 250000 \
#     --num_online_steps 0 \
#     --utd 1 \
#     --save_dir ./checkpoints/humanoid/wsrl 

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#     --exp_name NR5-humanoid-wsrl-ablation-iql-pretraining-seed-1 \
#     --env mujoco/humanoid/medium-v0 \
#     --agent iql \
#     --seed 1 \
#     --num_offline_steps 250000 \
#     --num_online_steps 0 \
#     --utd 1 \
#     --save_dir ./checkpoints/humanoid/wsrl 

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#     --exp_name NR5-humanoid-wsrl-ablation-iql-pretraining-seed-2 \
#     --env mujoco/humanoid/medium-v0 \
#     --agent iql \
#     --seed 2 \
#     --num_offline_steps 250000 \
#     --num_online_steps 0 \
#     --utd 1 \
#     --save_dir ./checkpoints/humanoid/wsrl 

#CalQL Pretraining

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#     --exp_name NR5-humanoid-wsrl-ablation-CalQL-pretraining-seed-0 \
#     --env mujoco/humanoid/medium-v0 \
#     --agent calql \
#     --seed 0 \
#     --num_offline_steps 250000 \
#     --num_online_steps 0 \
#     --utd 4 \
#     --save_dir ./checkpoints/humanoid/wsrl 

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#     --exp_name NR5-humanoid-wsrl-ablation-CalQL-pretraining-seed-1 \
#     --env mujoco/humanoid/medium-v0 \
#     --agent calql \
#     --seed 1 \
#     --num_offline_steps 250000 \
#     --num_online_steps 0 \
#     --utd 4 \
#     --save_dir ./checkpoints/humanoid/wsrl 

# bash experiments/scripts/locomotion/launch_wsrl_finetune.sh \
#     --exp_name NR5-humanoid-wsrl-ablation-CalQL-pretraining-seed-2 \
#     --env mujoco/humanoid/medium-v0 \
#     --agent calql \
#     --seed 2 \
#     --num_offline_steps 250000 \
#     --num_online_steps 0 \
#     --utd 4 \
#     --save_dir ./checkpoints/humanoid/wsrl


# deactivate virtual environment
deactivate

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";


# command to run
# sbatch --export=ALL,DATASET=data/SwedishMotorInsurance.csv, TARGET=Payment job.slurm
