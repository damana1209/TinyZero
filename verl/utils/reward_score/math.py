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
import random
import re
from typing import Tuple, Final
from enum import Enum


class MathStatus(Enum):
    BAD_FORMAT = 0
    WRONG_ANS_GOOD_FORMAT = 1
    RIGHT = 2
    IDK = 3


def _ORIGINAL_extract_solution(solution_str):
    # ? this is the original extract solution but does not fit the prompt that we use (from `math_dataset`)
    # ? "Let's think step by step and output the final answer within \\boxed{}. If you're unsure of how to solve the problem, just say \\boxed{I don't know}"
    """Check if the solution string is in a valid format."""

    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]

    if solution_str.count("<answer>") > 1:
        return None
    if "User:" in solution_str:
        return None
    if solution_str.count("<think>") > 1 or solution_str.count("</think>") != 1:
        return None

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


# def compute_score(solution_str, ground_truth) -> float:
#     retval = 0.0
#     to_print = random.randint(1, 64) == 1
#     if to_print:
#         print(f"Solution string: {solution_str}")
#     try:
#         extracted_solution = extract_solution(solution_str)
#         if extracted_solution is None:
#             retval = 0.0
#             return retval, MathStatus.BAD_FORMAT

#         string_in_last_boxed = last_boxed_only_string(extracted_solution)
#         if string_in_last_boxed is not None:
#             answer = remove_boxed(string_in_last_boxed)
#             if is_equiv(answer, ground_truth):
#                 retval = 1.0
#                 return retval, MathStatus.RIGHT
#             else:
#                 retval = 0.1  # format reward
#                 return retval, MathStatus.WRONG_ANS_GOOD_FORMAT_ANS_GOOD_FORMAT
#     except Exception as e:
#         print(e)

#     return retval, MathStatus.BAD_FORMAT


def extract_solution(solution_str: str) -> None | str:
    """
    solution_str: only the text returned by the LLM. ASSUMING IT IS RAW -- i.e. `\b` is the charecters `\` and `b` not backspace
    The string had "good" formatting if there is only one \\boxed{X} for some X.
    If the string is good, X is returned, else None (indicating bad formatting)
    """
    DELIMITER = r"\boxed{"
    par_mapping = {"(": ")", "{": "}", "[": "]", "<": ">"}

    # checked that there is only one answer
    if solution_str.rfind(DELIMITER) != solution_str.find(DELIMITER):
        return None

    else:
        del_idx = solution_str.find(DELIMITER)
        X = ""
        pars_stack: list = []

        if del_idx != -1:  # there was at least one instance
            for char in solution_str[del_idx + len(DELIMITER) :]:
                # is char open par?
                if char in par_mapping.keys():
                    pars_stack = pars_stack + [
                        char
                    ]  # push the open par to the top of the stack
                # is char close par?
                elif char in par_mapping.values():
                    # closing the first '{'
                    if len(pars_stack) == 0 and char == "}":
                        return X

                    # not closing any valid opening
                    elif len(pars_stack) == 0:
                        return None

                    # closing a valid par (pars_stack has 1 element due to the above check)
                    elif par_mapping[pars_stack[-1]] == char:
                        pars_stack.pop()

                # its just a regular char
                else:
                    pass

                X = X + str(char)

            # if we did not return from finding '}' then the DELIMITER was not properly closed
        # if we did not have 1 del_idx or we existed
        return None


# TODO move these into a global config
# TODO incorporate it in the MathStatus class above?
running_acc = 0.5
ema_coeff = 0.999
idk_min_reward = 0.1  # the real reward is the base accumlator
reward_dict = {
    MathStatus.RIGHT: 1,
    MathStatus.WRONG_ANS_GOOD_FORMAT: 0.1,
    MathStatus.BAD_FORMAT: 0,
}


def compute_score_idk_rs(
    solution_str: str, ground_truth: str, idk_max_reward=0.5
) -> dict:  # Tuple[float, Enum, float]:
    """
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
            ret: Final[dict] = {
                "reward_float": reward_dict[MathStatus.BAD_FORMAT],
                "reward_status_code": MathStatus.BAD_FORMAT,
                "cur_reward_for_idk": max(
                    idk_min_reward, min(running_acc, idk_max_reward)
                ),
            }  # ?make sure that the caller expctes a tuple

        elif extracted_solution == "I don't know":
            running_acc = ema_coeff * running_acc + (1 - ema_coeff) * idk_min_reward
            print(
                f"IDK detected, returning idk_reward: {min(running_acc, idk_max_reward)}, running_acc: {running_acc}"
            )
            ret: Final[dict] = {
                "reward_float": max(idk_min_reward, min(running_acc, idk_max_reward)),
                "reward_status_code": MathStatus.IDK,
                "cur_reward_for_idk": max(
                    idk_min_reward, min(running_acc, idk_max_reward)
                ),
            }  # ?make sure that the caller expctes a tuple

        # if extracted_solution is not None:
        elif is_equiv(extracted_solution, ground_truth):
            running_acc = (
                ema_coeff * running_acc
                + (1 - ema_coeff) * reward_dict[MathStatus.RIGHT]
            )
            ret: Final[dict] = {
                "reward_float": reward_dict[MathStatus.RIGHT],
                "reward_status_code": MathStatus.RIGHT,
                "cur_reward_for_idk": max(
                    idk_min_reward, min(running_acc, idk_max_reward)
                ),
            }  #
        # return val is wrong
        else:
            running_acc = (
                ema_coeff * running_acc
                + (1 - ema_coeff) * reward_dict[MathStatus.WRONG_ANS_GOOD_FORMAT]
            )
            ret: Final[dict] = {
                "reward_float": reward_dict[MathStatus.WRONG_ANS_GOOD_FORMAT],
                "reward_status_code": MathStatus.WRONG_ANS_GOOD_FORMAT,
                "cur_reward_for_idk": min(running_acc, idk_max_reward),
            }

    except Exception as e:
        print(e)
    to_print = random.randint(1, 64) == 1
    if to_print:
        print(
            f"OCCUSIONAL QUALITY PRINT: \n\n {solution_str=} \n {extracted_solution=} \n {ret=} \n {ground_truth=}"
        )

    return ret


def is_equiv(str1: str, str2: str, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


# def last_boxed_only_string(string: str):
#     idx = string.rfind("\\boxed")
#     if "\\boxed " in string:
#         return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
#     if idx < 0:
#         idx = string.rfind("\\fbox")
#         if idx < 0:
#             return None

#     i = idx
#     right_brace_idx = None
#     num_left_braces_open = 0
#     while i < len(string):
#         if string[i] == "{":
#             num_left_braces_open += 1
#         if string[i] == "}":
#             num_left_braces_open -= 1
#             if num_left_braces_open == 0:
#                 right_brace_idx = i
#                 break
#         i += 1

#     if right_brace_idx is None:
#         retval = None
#     else:
#         retval = string[idx : right_brace_idx + 1]

#     return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
