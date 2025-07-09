* `compute_score_idk_rs` is the reward shipping function
* `0.6.3.post2.dev0+ga2c71c540.d20250602.cu126` `vllm` version

// try on small models first

* make sure it runs on some exising dataset with the simpler reward function
* create a data preprocess for DAPO
* make sure it runs on the simple reward function
* run on the complex reward function
* 8G PUs
* <https://github.com/Zanette-Labs/efficient-reasoning/blob/main/run_rloo_7B.sh>
* use `uncertainty-verl` for this

--------------------

* refactor the return of the determine reward function
* refactor the reward manager to return generic metadata
* see who calls reward manager (in the training loop?) and also refactor validation to just print the metadata

currently `ray_trainer` is converting from raw numbers to single point metric data for the current "iteration." This makes sense, considering that it calls individual runs of detect reward.

* `_validate` is called 3 times in `ray_trainer.py` -- before after and during which makes sense
  * metrics are printed right after
* `reward_fn` is called once in `fit` and once in `_validate` but the outputs are handled in two different pieces of code?
  * `fit` calls `def compute_data_metrics(batch, use_critic=True):` and `_validate` processes them locally.
  * instead we should call `compute_data_metrics` from both places and use a `val` flag to handle the dictionary.
  * `RewardManager` should return the maximal amount of information (a dict) and we should process it based on the keys of the dictionary in `compute_data_metrics`
* how does `compute_data_metrics` expect its inputs?
* have 10 min -- next step is change the return from reward_fn
  * this is such a big change to the codebase and maybe I don't want to make it... at least it invalidates the other score function
