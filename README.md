# WSRL: Warm-Start Reinforcement Learning
[](https:/zhouzypaul.github.io/images/paper-images/wsrl/wsrl.png)
[![arXiv](https://img.shields.io/badge/arXiv-2412.07762-df2a2a.svg?style=for-the-badge)](http://arxiv.org/abs/2412.07762)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Static Badge](https://img.shields.io/badge/Project-Page-a?style=for-the-badge)](https://zhouzypaul.github.io/wsrl)

This is the code release for paper [Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data](http://arxiv.org/abs/2412.07762). We provide the implementation of [WSRL](http://arxiv.org/abs/2412.07762) (Warm-Start Reinforcement Learning), as well as popular actor-critic RL algorithms in JAX and Flax: [IQL](https://arxiv.org/abs/2110.06169), [CQL](https://arxiv.org/abs/2006.04779), [CalQL](https://arxiv.org/abs/2303.05479), [SAC](https://arxiv.org/abs/1801.01290), [RLPD](https://arxiv.org/abs/2302.02948). Variants of SAC also supported, such as [TD3](https://arxiv.org/pdf/1802.09477), [REDQ](https://arxiv.org/abs/2101.05982), and IQL policy extraction supports both AWR and DDPG+BC.
We support the following environments: D4RL antmaze, adroit, kitchen, and Mujoco locomotion, but the code can be easily adpated to work with other environments and datasets.

The code for the Franka robot experiments is located at [wsrl-robot](https://github.com/zhouzypaul/wsrl-robot). See running instructions in that repo.

![teaser](https://zhouzypaul.github.io/images/paper-images/wsrl/teaser.png)

```
@article{zhou2024efficient,
  author       = {Zhiyuan Zhou and Andy Peng and Qiyang Li and Sergey Levine and Aviral Kumar},
  title        = {Efficient Online Reinforcement Learning Fine-Tuning Need Not Retain Offline Data},
  conference   = {arXiv Pre-print},
  year         = {2024},
  url          = {http://arxiv.org/abs/2412.07762},
}
```


## Installation
```bash
conda create -n wsrl python=3.10 -y
conda activate wsrl
pip install -r requirements.txt
```

For jax, install
```
pip install --upgrade "jax[cuda11_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

To use the D4RL envs, you would also need my fork of the d4rl envs below.
This fork incorporates the antmaze-ultra environments and fixes the kitchen environment rewards to be consistent between the offline dataset and the environment.
```
git clone git@github.com:zhouzypaul/D4RL.git
cd D4RL
pip install -e .
```

To use Mujoco, you would also need to install mujoco manually to `~/.mujoco/` (for more instructions on download see [here](https://github.com/openai/mujoco-py?tab=readme-ov-file#install-mujoco)), and use the following environment variables
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
```

To use the adroit envs, you would need
```
git clone --recursive https://github.com/nakamotoo/mj_envs.git
cd mj_envs
git submodule update --remote
pip install -e .
```

Download the adroit dataset from [here](https://drive.google.com/file/d/1yUdJnGgYit94X_AvV6JJP5Y3Lx2JF30Y/view) and unzip the files into `~/adroit_data/`.
If you would like to put the adroit datasets into another directory, use the environment variable `DATA_DIR_PREFIX` (checkout the code [here](https://github.com/zhouzypaul/wsrl/blob/4b5665987079934a926c10a09bd81bc3c48ea9fa/wsrl/envs/adroit_binary_dataset.py#L7) for more details).
```bash
export DATA_DIR_PREFIX=/path/to/your/data
```

## Running
The main run script is `finetune.py`. We provide bash scripts in `experiments/scripts/<ENV>` to train WSRL/IQL/CQL/CalQ/RLPD on the different environments.

The shared agent configs are in `experiments/configs/*`, and the environment-specific configs are in `experiments/configs/train_config.py` and in the bash scripts.

### Pre-training
For example, to run CalQL (with Q-ensemble) pre-training
```bash
# on antmaze
bash experiments/scripts/antmaze/launch_calql_finetune.sh --use_redq --env antmaze-large-diverse-v2

# on adroit
bash experiments/scripts/adroit/launch_calql_finetune.sh --use_redq --env door-binary-v0

# on kitchen
bash experiments/scripts/kitchen/launch_calql_finetune.sh --use_redq --env kitchen-mixed-v0

# on mujoco locomotion (CQL pre-train because MC returns are hard to estimate)
bash experiments/scripts/locomotion/launch_cql_finetune.sh --use_redq --env halfcheetah-medium-replay-v0
```

### Fine-tuning
To run WSRL fine-tuning from a pre-trained checkpoint
```bash
# on antmaze
bash experiments/scripts/antmaze/launch_wsrl_finetune.sh --env antmaze-large-diverse-v2 --resume_path /path/to/checkpoint

# on adroit
bash experiments/scripts/adroit/launch_wsrl_finetune.sh --env door-binary-v0 --resume_path /path/to/checkpoint

# on kitchen
bash experiments/scripts/kitchen/launch_wsrl_finetune.sh --env kitchen-mixed-v0 --resume_path /path/to/checkpoint

# on mujoco locomotion
bash experiments/scripts/locomotion/launch_wsrl_finetune.sh --env halfcheetah-medium-replay-v0 --resume_path /path/to/checkpoint
```

### No Data Retention
The default setting is to not retain offline data during fine-tuning, as described in the [paper](http://arxiv.org/abs/2412.07762). However, if you wish to retain the data, you can use the `--offline_data_ratio <>` or `--online_sampling_method append` option. Checkout `finetune.py` for more details.

## Contributing
For a detailed explanation of how the codebase works, please checkout the [contributing.md](contributing.md) file.

To enable code checks and auto-formatting, please install pre-commit hooks (run this in the root directory):
```
pre-commit install
```
The hooks should now run before every commit. If files are modified during the checks, you'll need to re-stage them and commit again.

## Credits
This repo is built upon a version of Dibya Ghosh's [jaxrl_minimal](https://github.com/dibyaghosh/jaxrl_minimal) repository, which also included contributions from Kevin Black, Homer Walke, Kyle Stachowicz, and others.
