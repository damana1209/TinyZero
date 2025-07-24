# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
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
"""
Preprocess the GSM8k dataset to parquet format
"""

import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":
    flavors_to_data_source = {
        "plain": "lighteval/MATH",
        "reward_for_idk": "lighteval/MATH_idk",
        "bestguess_and_uncertainty_est": "lighteval/MATH_bestguess_and_uncertainty_est",
        "RLCR": "lighteval/MATH_RLCR",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/math/")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument(
        "--which-flavor",
        choices=flavors_to_data_source.keys(),
        required=True,
        help="We will be using the math dataset, but the system prompt and flavor of evaluation varies.",
    )

    args = parser.parse_args()

    dataset = datasets.load_dataset("di-zhang-fdu/MATH12000", trust_remote_code=True)

    train_dataset = dataset["train"]

    test_dataset = datasets.load_dataset("di-zhang-fdu/MATH500", trust_remote_code=True)
    test_dataset = test_dataset["test"]

    # if parser.which_flavor == :
    #     instruction_following = (
    #         "Let's think step by step and output the final answer within \\boxed{}."
    #     )
    # else:
    #     instruction_following = "Let's think step by step and output the final answer within \\boxed{}. If you're unsure of how to solve the problem, just say \\boxed{I don't know}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split, data_source):
        def process_fn(example, idx):
            question = example.pop("problem")

            if args.which_flavor == "plain":
                question = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question} Show your work in <think> </think> tags and return the final answer in <answer> </answer> tags, for example <answer> \\boxed{{\\frac{{4}}{{5}}}} </answer>.
Assistant: Let me solve this step by step.
<think>"""
            elif args.which_flavor == "reward_for_idk":
                question = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: {question}

Show your work in <think> </think> tags and return the final answer in <answer> </answer> tags, for example 

<think> ... thinking process here ... </think>
<answer> 
$\\boxed{{\\frac{{4}}{{5}}}}$
</answer>.

If you get stuck or think that you've made a mistake, just say 

<think> ... thinking process here ... </think>
<answer> 
\\boxed{{I don't know}}
</answer>.

A: Let me solve this step by step.
<think>"""
            elif args.which_flavor == "bestguess_and_uncertainty_est":
                # ? i removed the <answer> </answer> tags becuase I don't reward the model for them and it doesn't seem to use them. I know they were used in the R1 paper...
                question = f"""This is a conversation between a User and an Assistant. The User asks the Assistant a question, and the Assistant solves it. Before providing final answers, the careful Assistant develops its solution step-by-step and evaluates its epistemic uncertainty about its response to not mislead the user. As it thinks, it notes where it might be uncertain. 

REQUIREMENTS: 
1) throughly think about the problem, step-by-step, noting steps you are unsure about. 
2) write your final, best guess for the final answer in \\boxed{{your\_best\_guess\_here}}.
3) review the question, your thinking process, and final answer to deterime your final uncertainty.
4) write "After careful consideration of the question, my thinking, and my final answer, I estimate the probability that my answer is correct is \\uncertainty{{\your\_uncertainty\_here}}." \your\_uncertainty\_here MUST be an a number between 0 and 1.

GRADING RUBRIC:
0 points for bad format. 
REWARD:
    IS_GOOD * IS_GOOD_FORMAT_REWARD +
    IS_CORRECT * IS_CORRECT_REWARD +
    (IS_CORRECT - \your\_uncertainty\_here) * IDK_REWARD +
    min (1 - \your\_uncertainty\_here, \your\_uncertainty\_here) * NOT_EXTREME_REWARD
What the one before last summand means the Assistant's uncertainty should be inversly correlated with whether it is correct. I.e. if you are right the uncertainty should be close to 0 and if you're wrong the uncertainty it should be close to 1.
The last summand is so to encourage the assistant to slightly hedge its bets. 

CONVERSATION BEGINS:

U: {question}

A: Let me solve this step by step.
"""
            elif args.which_flavor == "RLCR":
                # ? i removed the <answer> </answer> tags becuase I don't reward the model for them and it doesn't seem to use them. I know they were used in the R1 paper...
                question = f"""A conversation between User and Assistant. The user asks a question, and the Assistant
solves it. The assistant first thinks about the reasoning process in the mind, provides the user
with the final answer, then analyzes its confidence about the solution and then provides the
user with its confidence level. The confidence level is a number between 0 and 1 (inclusive)
enclosed within <confidence> </confidence> tags. The final answer is enclosed between
<answer> </answer> tags. The analysis about confidence and uncertainty is enclosed within
<analysis> </analysis> tags. The assistant should reason about its confidence in the
solution and its uncertainty in the solution within these tags. Here are some guidelines for the
analysis: 
1. Your task is to point out things where the model could be wrong in its thinking,
or things where there might be ambiguity in the solution steps, or in the reasoning process
itself.
2. You should not suggest ways of fixing the response, your job is only to reason about
uncertainties.
3. For some questions, the response might be correct. In these cases, It is also okay to have
only a small number of uncertainties and then explicitly say that I am unable to spot more
uncertainties.
4. Uncertainties might be different from errors. For example, uncertainties may arise from
ambiguities in the question, or from the application of a particular lemma/proof.
5. If there are alternate potential approaches that may lead to different answers, you should
mention them.
6. List out plausible uncertainties, do not make generic statements, be as specific about
uncertainties as possible.
7. Enclose this uncertainty analysis within <analysis> </analysis> tags.
The final format that must be followed is : <think> reasoning process here
</think> <answer> final answer here </analysis> <analysis> analysis about confidence
and uncertainty here </analysis> <confidence> confidence level here (number between 0
and 1) </confidence> )

User: {question}

Assistant: 
"""

            else:
                raise ValueError("this code witll never be reaced")


            # Create the data structure
            answer = example.pop("solution")
            solution = extract_solution(answer)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(
        function=make_map_fn("train", flavors_to_data_source[args.which_flavor]),
        with_indices=True,
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test", flavors_to_data_source[args.which_flavor]),
        with_indices=True,
    )

    print("Example:")
    print(train_dataset[0]["prompt"][0]["content"])
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset.to_parquet(
        os.path.join(local_dir, args.which_flavor, "train.parquet")
    )
    test_dataset.to_parquet(os.path.join(local_dir, args.which_flavor, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
