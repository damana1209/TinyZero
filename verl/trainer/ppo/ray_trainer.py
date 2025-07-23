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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

from collections import defaultdict
import math
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict

from verl.utils.reward_score.countdown import CountdownStatus
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayResourcePool,
    RayWorkerGroup,
    RayClassWithInitArgs,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)

# ? matan
from verl.utils.reward_score.math import MathStatus

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=resource_pool_name,
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(
    data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"
):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores: torch.Tensor = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if "ref_log_prob" in data.batch.keys():
        kld = core_algos.kl_penalty(
            data.batch["old_log_probs"],
            data.batch["ref_log_prob"],
            kl_penalty=kl_penalty,
        )  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(token_level_scores, dtype=torch.float32)

    # print(
    #     f"MATAN SIZE: {responses.shape=} {token_level_scores.shape=}, {data.batch['old_log_probs'].shape=} {data.batch['ref_log_prob'].shape=}, {kld.shape=}"
    # )
    #! below was my original comment and it is WRONG. token_level_rewards should only have a reward at the last token position -- I wrote this before I understoof the gae mechanism. Wait, but isn't token level scores already that size?
    # ?what shape do we expect here? we are going to have a reward for every position that we train, which in this case is 1024 (the max) so
    # breakpoint()
    token_level_rewards = (
        token_level_scores.expand((token_level_scores.shape[0], kld.shape[-1]))
        - beta * kld
    )

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    # ? this is not really kl. it is an approximation logprob - reflogprob (see `compute_kl`)
    # if current_kl < 0:
    #     breakpoint()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"critic/kl": current_kl, "critic/kl_coeff": beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == "gae":
        # print("MATAN: not expecting gae to be running")
        values = data.batch["values"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch["token_level_rewards"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=token_level_rewards,
            values=values,
            eos_mask=response_mask,
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == "grpo":
        token_level_rewards = data.batch["token_level_rewards"]
        index = data.non_tensor_batch["uid"]
        responses = data.batch["responses"]
        response_length = responses.size(-1)
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, index=index
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def add_metric_with_description(
    metrics_dict, descriptions_dict, key, value, description
):
    """
    Helper function to add a metric with its description.

    Args:
        metrics_dict: Dictionary to store metric values
        descriptions_dict: Dictionary to store metric descriptions
        key: Metric name
        value: Metric value
        description: Human-readable description of the metric
    """
    metrics_dict[key] = value
    descriptions_dict[key] = description


def compute_data_metrics(batch, use_critic=True):
    """
    the standard function to compute metrics from all the data returned from a single PPO iteration.

    I wanted to make this also callable from `_validate` but decided against it becuase the metrics computed are really different.
    Any computation from the data returned by `reward_fn` (of `RewardManager` type) should be processed in this function.
    `RewardManager` takes in the raw token level responses and assignes each response a reward.
    logging is then taken care of by
    ```
    metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )

                logger.log(data=metrics, step=self.global_steps)
    ```
    """
    # MATAN: Debug attention mask issue
    # print("=== DEBUGGING ATTENTION MASK ===")
    # responses = batch.batch["responses"]

    # # Check token IDs
    # pad_token_id = 151643  # Based on your findings
    # print(f"Pad token ID: {pad_token_id}")

    # # Get the corrected response info
    # response_info = _compute_response_info(batch)
    # corrected_response_mask = response_info["response_mask"]

    # # Check first few responses
    # for i in [0, 1]:
    #     response = responses[i]
    #     print(f"\nResponse {i}:")

    #     # Find pad positions
    #     pad_positions = (response == pad_token_id).nonzero().flatten()
    #     if len(pad_positions) > 0:
    #         first_pad = pad_positions[0].item()
    #         print(f"  First pad at: {first_pad}")
    #         print(f"  Content length: {first_pad}")
    #     else:
    #         print(f"  No padding found, full length: {response.shape[0]}")

    #     # Check corrected response mask
    #     corrected_valid_length = corrected_response_mask[i].sum().item()
    #     print(f"  Corrected response mask says valid length: {corrected_valid_length}")
    #     print(
    #         f"  Corrected mask last 10 values: {corrected_response_mask[i][-10:].tolist()}"
    #     )

    #     # Verify fix worked
    #     if len(pad_positions) > 0:
    #         expected_valid = pad_positions[0].item()
    #         if expected_valid == corrected_valid_length:
    #             print(
    #                 f"  ✓ FIX WORKED! Expected: {expected_valid}, Got: {corrected_valid_length}"
    #             )
    #         else:
    #             print(
    #                 f"  ✗ FIX FAILED! Expected: {expected_valid}, Got: {corrected_valid_length}"
    #             )
    # print("=== END DEBUG ===\n")

    # assert False, "I wrote `_matan_data_metrics` because I had relatively low confidence in "
    # TODO: add response length --CHANGE THIS FUNCTION TO THE WAY THESE ARE REFERED TO
    sequence_score: torch.Tensor = (
        batch.batch["token_level_scores"]
        .sum(-1)
        .to(
            dtype=torch.bfloat16
        )  # got some error that was using long -- converted to float
    )
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    status_list = batch.non_tensor_batch["statuses"]
    # ? figure out how these come about
    rewards_for_idk_list = batch.non_tensor_batch["idk_rewards"].flatten().mean()

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]
    # Count number of 0.5s in sequence_score and calculate average
    # idks = torch.logical_and(sequence_score > 0.2, sequence_score < 0.9).float().mean().item()
    idk_ratio = (
        np.count_nonzero(status_list == MathStatus.IDK) / len(status_list)
        if len(status_list) > 0
        else 0.0
    )
    correct_ratio = (
        np.count_nonzero(status_list == MathStatus.RIGHT) / len(status_list)
        if len(status_list) > 0
        else 0.0
    )
    wrong_ans_good_format_ratio = (
        np.count_nonzero(status_list == MathStatus.WRONG_ANS_GOOD_FORMAT)
        / len(status_list)
        if len(status_list) > 0
        else 0.0
    )
    bad_format_ratio = (
        np.count_nonzero(status_list == MathStatus.BAD_FORMAT) / len(status_list)
        if len(status_list) > 0
        else 0.0
    )
    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    # TODO replace this with a tensorized method
    # prompt mask is left-padded, so starts with False and then turns true
    prompt_length = torch.tensor(
        [
            prompt_mask.shape[1] - (prompt_mask[idx, :] == 1).nonzero()[0]
            if (prompt_mask[idx, :] == 0).any()
            else prompt_mask.shape[1]
            for idx in range(prompt_mask.shape[0])
        ]
    ).to(dtype=torch.float32)

    response_length = torch.tensor(
        [
            (response_mask[idx, :] == 0).nonzero()[0]
            if (response_mask[idx, :] == 0).any()
            else response_mask.shape[1]
            for idx in range(response_mask.shape[0])
        ]
    ).to(dtype=torch.float32)

    # response_info = _compute_response_info(batch)
    # prompt_length = response_info["prompt_length"]
    # response_length = response_info["response_length"]
    # response_mask = response_info["response_mask"].bool()  # Use the CORRECTED mask

    # max_prompt_length = prompt_mask.size(-1)

    # Use the corrected response_mask for masking operations
    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]

        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {}
    descriptions = {}

    # Add metrics with descriptions using helper function
    add_metric_with_description(
        metrics,
        descriptions,
        "train/idk_ratio",
        idk_ratio,
        "Proportion of responses where model said 'I don't know'",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train/correct_ratio",
        correct_ratio,
        "Proportion of responses with correct mathematical answers",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train/wrong_ans_good_format_ratio",
        wrong_ans_good_format_ratio,
        "Proportion of responses with incorrect answers but valid format",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train/bad_format_ratio",
        bad_format_ratio,
        "Proportion of responses with invalid formatting",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train/avg_idk_reward",
        rewards_for_idk_list,
        "Average reward given for 'I don't know' responses",
    )

    # Score metrics
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/score/mean",
        torch.mean(sequence_score).detach().item(),
        "Mean sequence-level score across batch",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/score/max",
        torch.max(sequence_score).detach().item(),
        "Maximum sequence-level score in batch",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/score/min",
        torch.min(sequence_score).detach().item(),
        "Minimum sequence-level score in batch",
    )

    # Reward metrics
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/rewards/mean",
        torch.mean(sequence_reward).detach().item(),
        "Mean sequence-level reward (score + KL penalty) across batch",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/rewards/max",
        torch.max(sequence_reward).detach().item(),
        "Maximum sequence-level reward in batch",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/rewards/min",
        torch.min(sequence_reward).detach().item(),
        "Minimum sequence-level reward in batch",
    )

    # Advantage metrics
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/advantages/mean",
        torch.mean(valid_adv).detach().item(),
        "Mean advantage values for PPO policy gradient",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/advantages/max",
        torch.max(valid_adv).detach().item(),
        "Maximum advantage value in batch",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/advantages/min",
        torch.min(valid_adv).detach().item(),
        "Minimum advantage value in batch",
    )

    # Returns metrics
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/returns/mean",
        torch.mean(valid_returns).detach().item(),
        "Mean return values (cumulative future rewards)",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/returns/max",
        torch.max(valid_returns).detach().item(),
        "Maximum return value in batch",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "train-less-important/returns/min",
        torch.min(valid_returns).detach().item(),
        "Minimum return value in batch",
    )

    # Value function metrics (if using critic)
    if use_critic:
        add_metric_with_description(
            metrics,
            descriptions,
            "critic/values/mean",
            torch.mean(valid_values).detach().item(),
            "Mean value function predictions",
        )
        add_metric_with_description(
            metrics,
            descriptions,
            "critic/values/max",
            torch.max(valid_values).detach().item(),
            "Maximum value function prediction",
        )
        add_metric_with_description(
            metrics,
            descriptions,
            "critic/values/min",
            torch.min(valid_values).detach().item(),
            "Minimum value function prediction",
        )
        add_metric_with_description(
            metrics,
            descriptions,
            "critic/vf_explained_var",
            (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            "Value function explained variance (how well critic predicts returns)",
        )

    # Response length metrics
    add_metric_with_description(
        metrics,
        descriptions,
        "response_length/mean",
        torch.mean(response_length).detach().item(),
        "Average length of generated responses in tokens",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "response_length/max",
        torch.max(response_length).detach().item(),
        "Maximum response length in batch",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "response_length/min",
        torch.min(response_length).detach().item(),
        "Minimum response length in batch",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "response_length/clip_ratio",
        torch.mean(torch.eq(response_length, max_response_length).float())
        .detach()
        .item(),
        "Fraction of responses that hit maximum length limit",
    )

    # Prompt length metrics
    add_metric_with_description(
        metrics,
        descriptions,
        "prompt_length/mean",
        torch.mean(prompt_length).detach().item(),
        "Average length of input prompts in tokens",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "prompt_length/max",
        torch.max(prompt_length).detach().item(),
        "Maximum prompt length in batch",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "prompt_length/min",
        torch.min(prompt_length).detach().item(),
        "Minimum prompt length in batch",
    )
    add_metric_with_description(
        metrics,
        descriptions,
        "prompt_length/clip_ratio",
        torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
        "Fraction of prompts that hit maximum length limit",
    )

    # Store descriptions in metrics metadata for potential use by logger
    metrics["_descriptions"] = descriptions

    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{
            name: num_overall_tokens
            for name in ["ref", "values", "adv", "update_critic", "update_actor"]
        },
    }

    metrics = {}
    descriptions = {}

    # Add timing metrics with descriptions
    for name, value in timing_raw.items():
        add_metric_with_description(
            metrics,
            descriptions,
            f"timing_s/{name}",
            value,
            f"Wall-clock time in seconds for {name} phase",
        )

    # Add per-token timing metrics with descriptions
    for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys()):
        per_token_ms = timing_raw[name] * 1000 / num_tokens_of_section[name]
        add_metric_with_description(
            metrics,
            descriptions,
            f"timing_per_token_ms/{name}",
            per_token_ms,
            f"Time per token in milliseconds for {name} phase",
        )

    # Store descriptions in metrics metadata
    metrics["_descriptions"] = descriptions

    return metrics


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        reward_fn=None,
        val_reward_fn=None,  # TODO we should not allow this to be None
    ):
        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # ? wanted a new random seed each time, but to log to hf for reproducability -- haven't tried this
        # # Set global random seed if specified in config
        # if hasattr(self.config.trainer, "global_seed"):
        #     import torch
        #     import numpy as np
        #     import random

        #     global_seed = self.config.trainer.global_seed
        #     torch.manual_seed(global_seed)
        #     np.random.seed(global_seed)
        #     random.seed(global_seed)
        #     if torch.cuda.is_available():
        #         torch.cuda.manual_seed_all(global_seed)
        #     print(f"Set global random seed to: {global_seed}")

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == "fixed":
                self.kl_ctrl = core_algos.FixedKLController(
                    kl_coef=config.algorithm.kl_ctrl.kl_coef
                )
            elif config.algorithm.kl_ctrl.type == "adaptive":
                assert config.algorithm.kl_ctrl.horizon > 0, (
                    f"horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}"
                )
                self.kl_ctrl = core_algos.AdaptiveKLController(
                    init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                    target_kl=config.algorithm.kl_ctrl.target_kl,
                    horizon=config.algorithm.kl_ctrl.horizon,
                )
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.0)

        self._create_dataloader()

    def _create_dataloader(self):
        from torch.utils.data import DataLoader

        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        # breakpoint()
        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation=self.config.data.get("truncation", "error"),
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )
        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            prompt_key=self.config.data.prompt_key,
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation=self.config.data.get("truncation", "error"),
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset),
            shuffle=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1

        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps

    def _validate(self):
        val_reward_fn_returns = defaultdict(list)
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            """
            (Pdb) test_batch.batch.keys()
            _TensorDictKeysView(['prompts', 'attention_mask', 'input_ids', 'position_ids', 'responses'],
            (Pdb) test_batch.non_tensor_batch.keys()
            dict_keys(['answer', 'subject', 'level', 'unique_id', 'data_source', 'ability', 'reward_model', 'extra_info', 'index'])
            """
            # test_batch = test_batch.to('cuda')

            # we only do validation on rule-based rm
            # ? we are doing rule-based RM so we expect it to work
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                assert False, "not expecting whatever this is"
                return {}

            test_gen_batch = test_batch.pop(
                ["input_ids", "attention_mask", "position_ids"]
            )
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "validate": True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            # ?pad to the context?
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )
            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            # for certain reward function (e.g. sandbox), the generation can overlap with reward

            # TODO fix this bad pattern
            (val_reward_fn_return, _) = self.val_reward_fn(test_batch)

            for k, v in val_reward_fn_return.items():
                if isinstance(v, list):
                    val_reward_fn_returns[k].extend(v)
                else:
                    val_reward_fn_returns[k].append(v)
            # reward_tensor_lst.append(reward_tensor)
            # running_acc_lst.append(running_acc)
            # status_lst.extend(
            #     status_list

            # reward_tensor = (
            #     torch.cat(reward_tensor_lst, dim=0).sum(dim=-1).cpu()
            # )  # (batch_size,)
            # data_sources = np.concatenate(data_source_lst, axis=0)
            # # evaluate test_score based on data source
            # data_source_reward = {}
            # for i in range(reward_tensor.shape[0]):
            #     data_source = data_sources[i]
            #     if data_source not in data_source_reward:
            #         data_source_reward[data_source] = []
            #     data_source_reward[data_source].append(reward_tensor[i].item())
            metric_dict = {}
            # for data_source, rewards in data_source_reward.items():
            if "reward_status_code" in val_reward_fn_returns.keys():
                status_lst = np.array(
                    val_reward_fn_return["reward_status_code"], dtype=object
                )
                idk_ratio = (
                    np.count_nonzero(status_lst == MathStatus.IDK) / len(status_lst)
                    if len(status_lst) > 0
                    else 0.0
                )
                correct_ratio = (
                    np.count_nonzero(status_lst == MathStatus.RIGHT) / len(status_lst)
                    if len(status_lst) > 0
                    else 0.0
                )
                wrong_ratio = (
                    np.count_nonzero(status_lst == MathStatus.WRONG_ANS_GOOD_FORMAT)
                    / len(status_lst)
                    if len(status_lst) > 0
                    else 0.0
                )
                bad_format = (
                    np.count_nonzero(status_lst == MathStatus.BAD_FORMAT)
                    / len(status_lst)
                    if len(status_lst) > 0
                    else 0.0
                )

                metric_dict[f"val/correct_ratio"] = correct_ratio
                metric_dict[f"val/wrong_ratio"] = wrong_ratio
                metric_dict[f"val/idk_ratio"] = idk_ratio
                metric_dict[f"val/bad_format_ratio"] = bad_format
            else:
                assert False, (
                    "I am expecting 'reward_status_code' to be a key returned by `val_reward_fn`"
                )

            if "reward_float" in val_reward_fn_returns.keys():
                metric_dict[f"val/avg_reward"] = sum(
                    val_reward_fn_returns["reward_float"]
                ) / len(val_reward_fn_returns["reward_float"])
            else:
                assert False, "expecting `reward_float` in val_reward_fn"

            if "cur_reward_for_idk" in val_reward_fn_returns.keys():
                metric_dict[f"val/avg_reward_for_dk"] = sum(
                    val_reward_fn_returns["cur_reward_for_idk"]
                ) / len(val_reward_fn_returns["cur_reward_for_idk"])
            else:
                assert False, "expecting `avg_reward_for_idk` in `val_reward_fn`"

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = (
                actor_rollout_cls
            )
        else:
            raise NotImplementedError

        # create critic
        # MATAN: Assert that we're not using GAE since we don't expect to need a critic for math dataset
        # ? the warning below is wrong -- its okay to use gae and ppo. (I also do now see how grpo can be seen as ppo with a larger per-prompt batch and a different advantage estimator, at least sort of :)
        # if self.config.algorithm.adv_estimator == "gae":
        #     print(f"ERROR: Using GAE advantage estimator which requires a critic!")
        #     print(
        #         f"For math dataset with ground truth answers, consider using GRPO instead:"
        #     )
        #     print(f"Add 'algorithm.adv_estimator=grpo \\' to your training script")
        #     print(f"Current adv_estimator: {self.config.algorithm.adv_estimator}")
        # assert False, (
        #     f"Unexpected use of GAE advantage estimator. Expected GRPO for math dataset."
        # )

        if self.config.algorithm.adv_estimator == "gae":
            # breakpoint()
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
            self.use_critic = True
        elif self.config.algorithm.adv_estimator == "grpo":
            self.use_critic = False
        else:
            raise NotImplementedError

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            assert False, "not expecting to be using reward models at the moment"
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        # breakpoint()
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            assert False, "not expecting to use reward model at the moment"
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        actor_local_path = os.path.join(
            self.config.trainer.default_local_dir,
            "actor",
            f"global_step_{self.global_steps}",
        )
        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, "actor")
        )
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path)

        if self.use_critic:
            critic_local_path = os.path.join(
                self.config.trainer.default_local_dir,
                "critic",
                f"global_step_{self.global_steps}",
            )
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, "critic")
            )
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path)

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """
        Reorder the data on the single controller so that each data parallel (dp) rank receives a similar total number of tokens.

        What is a dp rank?
        - "dp rank" refers to the index of a process or worker in a data parallel group. In distributed training, data parallelism (DP) means splitting the input batch across multiple workers (ranks), each of which processes a portion of the data and computes gradients independently. These gradients are then synchronized across all ranks.
        - Each "dp rank" is responsible for a shard of the batch, and balancing the number of tokens per rank helps ensure efficient and fair workload distribution.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        )  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        ?I may have broken this code's abillity to process multiple datasources
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # # Log metric descriptions once at the beginning of training
        # metric_descriptions = {
        #     "train/idk_ratio": "Proportion of responses where model said 'I don't know'",
        #     "train/correct_ratio": "Proportion of responses with correct mathematical answers",
        #     "train/wrong_ans_good_format_ratio": "Proportion of responses with incorrect answers but valid format",
        #     "train/bad_format_ratio": "Proportion of responses with invalid formatting",
        #     "train/avg_idk_reward": "Average reward given for 'I don't know' responses",
        #     "train-less-important/score/mean": "Mean sequence-level score across batch",
        #     "train-less-important/rewards/mean": "Mean sequence-level reward (score + KL penalty) across batch",
        #     "train-less-important/advantages/mean": "Mean advantage values for PPO policy gradient",
        #     "train-less-important/returns/mean": "Mean return values (cumulative future rewards)",
        #     "response_length/mean": "Average length of generated responses in tokens",
        #     "response_length/clip_ratio": "Fraction of responses that hit maximum length limit",
        #     "prompt_length/mean": "Average length of input prompts in tokens",
        #     # Add more descriptions as needed...
        # }

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # we start from step 1
        self.global_steps += 1

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                print(f"epoch {epoch}, step {self.global_steps}")
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # breakpoint()
                # pop those keys for generation
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"]
                )

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch
                        )

                    # MATAN: Debug generation output
                    # print("=== POST-GENERATION DEBUG ===")
                    # responses = gen_batch_output.batch["responses"]
                    # attention_mask = gen_batch_output.batch["attention_mask"]
                    # print(f"Generated attention mask shape: {attention_mask.shape}")
                    # print(f"Generated responses shape: {responses.shape}")

                    # # Check a few samples
                    # for i in [0, 1]:
                    #     response = responses[i]
                    #     resp_mask = attention_mask[i, -responses.shape[1] :]

                    #     # Find pad tokens (assumed to be 151643 based on our debugging)
                    #     pad_positions = (response == 151643).nonzero().flatten()
                    #     if len(pad_positions) > 0:
                    #         first_pad = pad_positions[0].item()
                    #         print(f"  Response {i}: First pad at {first_pad}")
                    #         print(
                    #             f"  Response {i}: Attention mask valid length: {resp_mask.sum().item()}"
                    #         )
                    #         print(
                    #             f"  Response {i}: Last 10 attention values: {resp_mask[-10:].tolist()}"
                    #         )
                    #         if first_pad != resp_mask.sum().item():
                    #             print(
                    #                 f"  *** GENERATION BUG: Expected {first_pad}, got {resp_mask.sum().item()} ***"
                    #             )
                    # print("=== END POST-GENERATION DEBUG ===")

                    # ? I think the following three assignement statements are only relavent when we we generate several responses for the same prompt (i.e. actor_rollout_ref.rollout.n > 1) we are currently not doing this.
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n,
                        interleave=True,
                    )
                    batch = batch.union(gen_batch_output)

                    # balance the number of valid tokens on each dp rank.
                    #! Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    # ? what is this?
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    if self.use_reference_policy:
                        # we do expect to use kl regularization which requires reference_policy
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # breakpoint()
                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)
                        # if self.global_steps ==1:
                        #     reward_fn_ret, logger.wandb.config = self.reward_fn(batch)

                        # else
                        # TODO set wandb init here to the params returned by the reward fn
                        (reward_fn_ret, _) = self.reward_fn(batch)

                        # ? unpack the return of the reward fn on the batch (calls RewardManager which calls )
                        if "reward_float" in reward_fn_ret.keys():
                            # ? I think this is actually a pretty reasonable pattern? The reward fn returns the information in the 'plainest' way possible, and the trainer converts it to its desired format
                            temp = torch.zeros_like(batch.batch["responses"])
                            for i in range(batch.batch["responses"].shape[0]):
                                # ? I'm kinda weirded out by the fact that valid_response_length is always 1024 -- shouldn't it be like... eh...
                                temp[
                                    i, reward_fn_ret["valid_response_length"][i] - 1
                                ] = reward_fn_ret["reward_float"][i]
                            batch.batch["token_level_scores"] = temp

                        if "reward_status_code" in reward_fn_ret.keys():
                            batch.non_tensor_batch["statuses"] = np.array(
                                reward_fn_ret["reward_status_code"], dtype=object
                            )
                        if "cur_reward_for_idk" in reward_fn_ret.keys():
                            batch.non_tensor_batch["idk_rewards"] = np.array(
                                reward_fn_ret["cur_reward_for_idk"], dtype=object
                            )

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.use_kl_loss:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                        )
                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # assert False, "not expecting to use critic"
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate in training loop
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and self.global_steps % self.config.trainer.test_freq == 0
                    ):
                        with _timer("validation", timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)
                        # ? what do we do with the metrics once updated? ok, we finally log the metrics for the entire iteration -- so that's good

                    if (
                        self.config.trainer.save_freq > 0
                        and self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # ? we've collected metrics into this large dictionary. some metrics needed to be computed from the logged data metrics and we do that here
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )

                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:
                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f"Final validation metrics: {val_metrics}")
                        logger.log(data=val_metrics, step=self.global_steps)
                    return
