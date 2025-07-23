name = "bestguess_and_uncertainty_est.py"
description = """REWARD:
    formatting_reward + 
    IS_GOOD_FORMATTING * formatting_reward +
    IS_CORRECT * IS_CORRECT_REcorrectness_reward_multiplierWARD +
    (IS_CORRECT - uncertainty) * calibration_reward_multiplier +
    min (1 - uncertainty, uncertainty) * hedging_reward_multipler"""
formatting_reward = 0.1
calibration_reward_multiplier = 1
hedging_reward_multipler = 0.2
correctness_reward_multiplier = 1

everything_dict = {
    "name": name,  # ? do not have 2 different reward fns with the same name...
    "description": description,
    "params": {
        "formatting_reward": formatting_reward,
        "calibration_reward_multiplier": calibration_reward_multiplier,
        "hedging_reward_multipler": hedging_reward_multipler,
        "correctness_reward_multiplier": correctness_reward_multiplier,
    },
}


# def update_config(config: dict) -> dict:
#     assert config is not None

#     config["reward_fn"] = {
#         "name": name,
#         "description": description,
#         "params": {
#             "formatting_reward": formatting_reward,
#             "calibration_reward_multiplier": calibration_reward_multiplier,
#             "hedging_reward_multipler": hedging_reward_multipler,
#             "correctness_reward_multiplier": correctness_reward_multiplier,
#         },
#     }
#     return config
