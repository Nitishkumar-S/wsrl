from ml_collections import ConfigDict

from experiments.configs.cql_config import get_config as get_cql_config
from experiments.configs.iql_config import get_config as get_iql_config
from experiments.configs.sac_config import get_config as get_sac_config
from experiments.configs.wsrl_config import get_config as get_wsrl_config


def get_config(config_string):

    possible_structures = {

        ########################################################
        #                    antmaze configs                   #
        ########################################################

        "antmaze_cql": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="uniform",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [256, 256, 256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        cql_autotune_alpha=True,
                        cql_target_action_gap=0.8,
                    )
                ).to_dict(),
            )
        ),

        "antmaze_iql":ConfigDict(
            dict(
                agent_kwargs=get_iql_config(
                    updates=dict(
                        expectile=0.9,
                        temperature=10.0,
                    )
                ).to_dict(),
            )
        ),

        "antmaze_wsrl": ConfigDict(
            dict(
                agent_kwargs=get_wsrl_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="uniform",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [256, 256, 256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": True,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": True,
                        },
                        max_target_backup=True,
                    )
                ).to_dict(),
            )
        ),

        ########################################################
        #                    adroit configs                    #
        ########################################################

        "adroit_cql": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="exp",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "kernel_scale_final": 1e-2,
                            "activations": "relu",
                        },
                        policy_network_kwargs={
                            "hidden_dims": [512, 512],
                            "kernel_scale_final": 1e-2,
                            "activations": "relu",
                        },
                        online_cql_alpha=1.0,
                        cql_alpha=1.0,
                    )
                ).to_dict(),
            )
        ),

        "adroit_iql":ConfigDict(
            dict(
                agent_kwargs=get_iql_config(
                    updates=dict(
                        policy_network_kwargs=dict(
                            hidden_dims=(256, 256),
                            kernel_init_type="var_scaling",
                            kernel_scale_final=1e-2,
                            dropout_rate=0.1,
                        ),
                        expectile=0.7,
                        temperature=0.5,
                    ),
                ).to_dict(),
            )
        ),

        "adroit_wsrl": ConfigDict(
            dict(
                agent_kwargs=get_wsrl_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="exp",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "kernel_scale_final": 1e-2,
                            "activations": "relu",
                            "use_layer_norm": True,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [512, 512],
                            "kernel_scale_final": 1e-2,
                            "activations": "relu",
                            "use_layer_norm": True,
                        },
                    )
                ).to_dict(),
            )
        ),

        ########################################################
        #                    kitchen configs                   #
        ########################################################

        "kitchen_cql": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="exp",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "activations": "relu",
                        },
                        policy_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "activations": "relu",
                        },
                        online_cql_alpha=5.0,
                        cql_alpha=5.0,
                        cql_importance_sample=False,
                    )
                ).to_dict(),
            )
        ),

        "kitchen_iql":ConfigDict(
            dict(
                agent_kwargs=get_iql_config(
                    updates=dict(
                        policy_network_kwargs=dict(
                            hidden_dims=(256, 256),
                            activations="relu",
                            dropout_rate=0.1,
                        ),
                        critic_network_kwargs=dict(
                            hidden_dims=(256, 256),
                            activations="relu",
                        ),
                        expectile=0.7,
                        temperature=0.5,
                    )
                ).to_dict(),
            )
        ),

        "kitchen_wsrl": ConfigDict(
            dict(
                agent_kwargs=get_wsrl_config(
                    updates=dict(
                        policy_kwargs=dict(
                            tanh_squash_distribution=True,
                            std_parameterization="exp",
                        ),
                        critic_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "activations": "relu",
                            "use_layer_norm": True,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [512, 512, 512],
                            "activations": "relu",
                            "use_layer_norm": True,
                        },
                    )
                ).to_dict(),
            )
        ),

        ########################################################
        #                  locomotion configs                  #
        ########################################################

        "locomotion_cql": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        critic_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        online_cql_alpha=5.0,
                        cql_alpha=5.0,
                    )
                ).to_dict(),
            )
        ),

        "locomotion_iql":ConfigDict(
            dict(
                agent_kwargs=get_iql_config(
                    updates=dict(
                        expectile=0.7,
                        temperature=3.0,
                    )
                ).to_dict(),
            )
        ),

        # WSRL on locomotion runs the `cql` agent, not `sac`, so this is the CQL
        # config plus WSRL's REDQ ensemble + layer norm (what get_wsrl_config adds
        # on top of sac_config). The other *_wsrl configs are still SAC-based.
        "locomotion_wsrl": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        critic_ensemble_size=10,
                        critic_subsample_size=2,
                        critic_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": True,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [256, 256],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": True,
                        },
                        online_cql_alpha=5.0,
                        cql_alpha=5.0,
                    )
                ).to_dict(),
            )
        ),

        ########################################################
        #        locomotion width ablation ([512, 512])        #
        ########################################################
        # Humanoid-inspired network-width ablation. Identical to the three
        # locomotion configs above except hidden_dims is [512, 512] instead of
        # [256, 256] in both the critic and the policy. Motivation: Humanoid has
        # a 348-dim observation vs halfcheetah's 17, which [256, 256] may
        # bottleneck. Select with --config .../train_config.py:locomotion_<algo>_wide.
        # These are separate entries so the baseline configs are untouched.

        "locomotion_cql_wide": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        critic_network_kwargs={
                            "hidden_dims": [512, 512],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [512, 512],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                        },
                        online_cql_alpha=5.0,
                        cql_alpha=5.0,
                    )
                ).to_dict(),
            )
        ),

        # locomotion_iql sets no network kwargs -- its (256, 256) comes from the
        # iql_config defaults. So the wide variant must respecify both network
        # kwargs (mirroring iql_config's other defaults) rather than only bumping
        # a number.
        "locomotion_iql_wide": ConfigDict(
            dict(
                agent_kwargs=get_iql_config(
                    updates=dict(
                        expectile=0.7,
                        temperature=3.0,
                        policy_network_kwargs=dict(
                            hidden_dims=(512, 512),
                            kernel_init_type="var_scaling",
                            kernel_scale_final=1e-2,
                        ),
                        critic_network_kwargs=dict(
                            hidden_dims=(512, 512),
                            kernel_init_type="var_scaling",
                        ),
                    )
                ).to_dict(),
            )
        ),

        "locomotion_wsrl_wide": ConfigDict(
            dict(
                agent_kwargs=get_cql_config(
                    updates=dict(
                        critic_ensemble_size=10,
                        critic_subsample_size=2,
                        critic_network_kwargs={
                            "hidden_dims": [512, 512],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": True,
                        },
                        policy_network_kwargs={
                            "hidden_dims": [512, 512],
                            "activations": "relu",
                            "kernel_scale_final": 1e-2,
                            "use_layer_norm": True,
                        },
                        online_cql_alpha=5.0,
                        cql_alpha=5.0,
                    )
                ).to_dict(),
            )
        ),
    }

    return possible_structures[config_string]
