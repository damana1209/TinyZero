# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/hendrycks_math/utils.py
from enum import IntEnum

from .reward_function_utils import (
    is_equiv,
    extract_solution_square_brackets,
    to_unit_interval,
    extract_solution_tags,
)
import torch

class MathStatus(IntEnum):
    BAD_FORMAT = 0
    WRONG_ANS_GOOD_FORMAT = 1
    RIGHT = 2
    IDK = 3

# TODO create a dict class which standardizes the return type
# TODO how do I enforece that each conditional returns the all the same keys? The current pattern does not do that.

def rlcr(solution_str: str, ground_truth: str) -> dict:
    """
    replicating the methods from https://www.arxiv.org/pdf/2507.16806
    """
    pass
    answer_think, i_answer_think = extract_solution_tags(solution_str, r"think")
    solution_str = solution_str[i_answer_think:]
    answer, i_answer = extract_solution_tags(solution_str, r"answer")
    solution_str = solution_str[i_answer:]
    confidence_think, i_confidence_think = extract_solution_tags(
        solution_str, r"analysis"
    )
    solution_str = solution_str[i_confidence_think:]
    confidence, _ = extract_solution_tags(solution_str, r"confidence")
    if None in [answer_think, answer, confidence_think, confidence]:
        return {
            "reward_float": 0,
            "reward_status_code": MathStatus.BAD_FORMAT,
            "confidence": torch.nan,
            "brier_score": torch.nan,
        }
    elif None in [to_unit_interval(confidence)]:
        return {
            "reward_float": 0,
            "reward_status_code": MathStatus.BAD_FORMAT,
            "confidence": torch.nan,
            "brier_score": torch.nan,
        }
    else:
        from reward_fn_data.RLCR import formatting_reward

        is_correct = is_equiv(answer, ground_truth)
        confidence_float = to_unit_interval(confidence)
        return {
            "reward_float": formatting_reward
            + is_correct
            - (confidence_float - is_correct) ** 2,
            "reward_status_code": (
                MathStatus.RIGHT if is_correct else MathStatus.WRONG_ANS_GOOD_FORMAT
            ),
            "confidence": confidence_float,
            "brier_score": (confidence_float - is_correct) ** 2,
        }


def bestguess_and_uncertainty_est(solution_str: str, ground_truth: str) -> dict:
    """
    ! remember to change the math_dataset with any formatting changes (need to tell the model)
    The assistant is encourged to think step by step and note its uncertainty. It is prompted to respond with both is 'best guess' and an 'uncertainty score' and is graded on being correct and calibrated (i.e. uncertainty inversly correlated with correctness). (see `math_dataset` for the complete prompt)
    FORMAT: we will check there is exactly one `\\boxed{your_best_guess_here}` and exactly one `After careful consideration of the question, my thinking, and my final answer, I estimate the probability that my answer is correct is \\uncertainty{{[0,1]]}` (o.w. bad format)
    REWARD:
    IS_GOOD * IS_GOOD_FORMAT_REWARD +
    IS_CORRECT * IS_CORRECT_REWARD +
    (IS_CORRECT - uncertainty) * IDK_REWARD +
    min (1 - uncertainty, uncertainty) * NOT_EXTREME_REWARD
    """
    from reward_fn_data.bestguess_and_uncertainty_est import (
        formatting_reward,
        calibration_reward_multiplier,
        hedging_reward_multipler,
        correctness_reward_multiplier,
    )

    best_guess = extract_solution_square_brackets(solution_str=solution_str)
    uncertainty: str = extract_solution_square_brackets(
        solution_str=solution_str,
        DELIMITER=r"\\uncertainty{",
    )
    uncertainty_float = to_unit_interval(uncertainty)
    # bad format
    if best_guess is None or uncertainty is None or uncertainty_float is None:
        return {
            "reward_float": 0,
            "reward_status_code": MathStatus.BAD_FORMAT,
        }
    else:
        is_correct = is_equiv(best_guess, ground_truth)
        correctness_reward = is_correct * correctness_reward_multiplier
        calibration_reward = (
            float(is_correct) - uncertainty_float
        ) * calibration_reward_multiplier
        hedging_reward = (
            min(1 - uncertainty_float, uncertainty_float) * hedging_reward_multipler
        )
        total_reward = (
            formatting_reward + correctness_reward + calibration_reward + hedging_reward
        )
        return {
            "reward_float": total_reward,
            "reward_status_code": (
                MathStatus.RIGHT if is_correct else MathStatus.WRONG_ANS_GOOD_FORMAT
            ),
            "uncertainty": uncertainty_float,
            "correctness_reward": correctness_reward,
            "calibration_reward": calibration_reward,
            "hedging_reward": hedging_reward,
            "formatting_reward": formatting_reward,
        }
        # TODO I could imagine better prints, but for now lets run.


###########################################
###########################################
###########################################


# TODO move these into a global config
# TODO incorporate it in the MathStatus class above?
idk_max_reward = 0.7
running_acc = 0.7
ema_coeff = 0.999
idk_min_reward = 0.1  # the real reward is the base accumlator
reward_dict = {
    MathStatus.RIGHT: 1,
    MathStatus.WRONG_ANS_GOOD_FORMAT: 0.1,
    MathStatus.BAD_FORMAT: 0,
}


def compute_score_idk_rs(
    solution_str: str, ground_truth: str
) -> dict:  # Tuple[float, Enum, float]:
    """
    #! turned off reward shipping for now by setting
    solution_str: only the model's response decoded to str
    ground_truth: the ground truth answer to the question given as a str
    idk_max_reward: the maximum reward we will give for idk (starts at running_acc as above and converges to the model current reward)

    Can return dict with a bunch of stuff and we check manually what keys are there -- required is `reward_float`.


    """
    # TODO check that our reward shipping makes sense
    global running_acc, ema_coeff
    try:
        extracted_solution = extract_solution(solution_str)
        if extracted_solution is None:
            ret = {
                "reward_float": reward_dict[MathStatus.BAD_FORMAT],
                "reward_status_code": MathStatus.BAD_FORMAT,
                "cur_reward_for_idk": max(
                    idk_min_reward, min(running_acc, idk_max_reward)
                ),
            }  # ?make sure that the caller expctes a tuple

        elif extracted_solution == "I don't know":
            running_acc = (
                0.7  # ema_coeff * running_acc + (1 - ema_coeff) * idk_min_reward
            )
            print(
                f"IDK detected, returning idk_reward: {min(running_acc, idk_max_reward)}, running_acc: {running_acc}"
            )
            ret = {
                "reward_float": max(idk_min_reward, min(running_acc, idk_max_reward)),
                "reward_status_code": MathStatus.IDK,
                "cur_reward_for_idk": max(
                    idk_min_reward, min(running_acc, idk_max_reward)
                ),
            }  # ?make sure that the caller expctes a tuple

        # if extracted_solution is not None:
        elif is_equiv(extracted_solution, ground_truth):
            running_acc = 0.7
            # (
            #     ema_coeff * running_acc
            #     + (1 - ema_coeff) * reward_dict[MathStatus.RIGHT]
            # )
            ret = {
                "reward_float": reward_dict[MathStatus.RIGHT],
                "reward_status_code": MathStatus.RIGHT,
                "cur_reward_for_idk": max(
                    idk_min_reward, min(running_acc, idk_max_reward)
                ),
            }  #
        # return val is wrong
        else:
            running_acc = 0.7  # (
            #     ema_coeff * running_acc
            #     + (1 - ema_coeff) * reward_dict[MathStatus.WRONG_ANS_GOOD_FORMAT]
            # )
            ret = {
                "reward_float": reward_dict[MathStatus.WRONG_ANS_GOOD_FORMAT],
                "reward_status_code": MathStatus.WRONG_ANS_GOOD_FORMAT,
                "cur_reward_for_idk": min(running_acc, idk_max_reward),
            }

    except Exception as e:
        print(e)
    return ret
