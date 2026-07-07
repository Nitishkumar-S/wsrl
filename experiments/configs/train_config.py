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

        "locomotion_wsrl": ConfigDict(
            dict(
                agent_kwargs=get_wsrl_config(
                    updates=dict(
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
                    )
                ).to_dict(),
            )
        ),
    }

    return possible_structures[config_string]
