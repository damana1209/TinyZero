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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from huggingface_hub.errors import UnknownError
from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, math, multiply, countdown
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import ray


# ?matan added
torch.set_default_dtype(torch.bfloat16)
torch.set_default_device("cuda")
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Optional, Tuple
from verl.utils.tracking import Tracking  # ? this is only for the type annotation

# REWARD_DATA_SOURCE_COMPTAB_DICT = {
#     "bestguess_and_uncertainty_est": ["lighteval/MATH_idk"]
# }
# # ? if this grows a lot we can add the functions into this dict and use it to map, instead of the if statement. For now this is not necessery

# ? can use this if several rewards for the same dataset
# def check_data_source_and_reward_are_compatible(
#     data_source: str, reward: Optional[str]
# ):
#     if reward is None:  # just pick the default reward for the data_source
#         return
#     elif data_source not in REWARD_DATA_SOURCE_COMPTAB_DICT[reward]:
#         raise ValueError(
#             f"Reward '{reward}' is incompatible with data_source '{data_source}'. "
#             f"Compatible data sources: {REWARD_DATA_SOURCE_COMPTAB_DICT[reward]}"
#         )


def _select_rm_score_fn(
    data_source: str, logger: Tracking | None
) -> Tuple[Callable[..., dict]]:
    """
    selects the reward function and returns an updated config which, where the key reward_fn corrospond to a dict of all the relavent values of the reward function
    """
    # old math
    if data_source == "openai/gsm8k":
        raise NotImplementedError(
            f"Matan : I may have deleted `math.compute_score`, the function which is supposed to process {data_source=}. I'd suggest looking at a previous version of the git history to recover this function (at a certain point it was there and did work!)"
        )
    elif data_source == "lighteval/MATH":
        raise NotImplementedError(
            f"Matan : I may have deleted `math.compute_score`, the function which is supposed to process {data_source=}. I'd suggest looking at a previous version of the git history to recover this function (at a certain point it was there and did work!)"
        )

    # new math
    elif data_source == "openai/gsm8k_idk":
        # return gsm8k.compute_score_idk
        return math.compute_score_idk_rs

    elif data_source == "lighteval/MATH_idk":
        return math.compute_score_idk_rs

    elif data_source == "lighteval/MATH_bestguess_and_uncertainty_est":
        if logger is not None:
            from reward_fn_data.bestguess_and_uncertainty_est import everything_dict

            logger.update_wandb_config_with_reward_fn(everything_dict)

        return math.bestguess_and_uncertainty_est

    # countdown
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown_idk_and_answer" in data_source:
        return countdown.compute_score_idk_and_answer
        # return countdown.compute_score_idk_and_answer_rs
    elif "countdown_idk" in data_source:
        return countdown.compute_score_idk_rs
    elif "countdown" in data_source:
        return countdown.compute_score
    else:
        raise NotImplementedError(f"{data_source}")


class RewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        # self.reward = reward

    def __call__(self, data: DataProto, logger=None) -> Tuple[dict, dict]:
        """
        Its annoying to do this in __call__ but the codebase wanted to initalize a rewardManager while being flexible on what data sources it can take an dI don't want to break it

        should return the "rawest data" and it will be processed to necessery form in `fit()`
        """

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            return data.batch["rm_scores"]

        already_print_data_sources = {}
        compute_score_fn_returns = defaultdict(list)
        # compute_score_fn_returns["reward_float"] = torch.zeros_like(
        #    data.batch["responses"], dtype=torch.float32
        # )

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][
                :prompt_length
            ].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][
                prompt_length:
            ].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            # ? previously was passing question (i.e. valid_prompt_ids) to the grader -- why?
            # ? maybe for some graders that is useful -- was kind of annoying for me
            concat_prompt_and_response_tokens = torch.cat(
                (valid_prompt_ids, valid_response_ids)
            )
            concat_prompt_and_response_str = self.tokenizer.decode(
                concat_prompt_and_response_tokens
            )

            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # select rm_score
            data_source = data_item.non_tensor_batch["data_source"]
            compute_score_fn = _select_rm_score_fn(data_source, logger)
            # breakpoint()
            compute_score_fn_return: dict = compute_score_fn(
                solution_str=response_str, ground_truth=ground_truth
            )
            assert isinstance(compute_score_fn_return, dict), (
                f"Matan: I changed the interface so instead of returning a tuple with varying number of params, we return a dict. Usually, each element of the dict will be an integer-ish type representing a particular metric of the response {type(compute_score_fn_return)=}"
            )
            # ? I will need to use the final reward to estimate advantage -- the previous approach was to already parse it as a tensor, I can instead
            for k, v in compute_score_fn_return.items():
                if isinstance(v, list):
                    compute_score_fn_returns[k].extend(v)
                else:
                    compute_score_fn_returns[k].append(v)

            compute_score_fn_returns["valid_response_length"].append(
                valid_response_length.item()
            )
            compute_score_fn_return["max_score_length"] = data.batch["responses"].shape[
                -1
            ]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                # assert False, (
                #     f"I don't know what's printing data sources? {self.num_examine=} {already_print_data_sources=} {data_source=}"
                # )
                already_print_data_sources[data_source] += 1
                print(concat_prompt_and_response_str)
        return compute_score_fn_returns


import ray
import hydra


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    if not ray.is_initialized():
        ray.init(
            dashboard_host="0.0.0.0",
            dashboard_port=8265,
            runtime_env={
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "RAY_DEBUG": "1",
                    "RAY_DEBUG_POST_MORTEM": "1",
                }
            },
        )
        # ?Daman's setting: ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN', "RAY_DEBUG": "legacy"}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf

    pprint(
        OmegaConf.to_container(config, resolve=True)
    )  # resolve=True will eval symbol values
    OmegaConf.resolve(config)
    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer

    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    # ? I am surprised that we currently don't seem to specify this -- I think we're using FSDP
    if config.actor_rollout_ref.actor.strategy == "fsdp":
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup

        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == "megatron":
        assert False, "expecting to use fsdp, not megatron"
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup

        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == "fsdp":
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(
        resource_pool_spec=resource_pool_spec, mapping=mapping
    )
    trainer = RayPPOTrainer(
        config=config,
        tokenizer=tokenizer,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        reward_fn=reward_fn,
        val_reward_fn=val_reward_fn,
    )
    trainer.init_workers()
    # breakpoint()
    trainer.fit()


if __name__ == "__main__":
    main()
