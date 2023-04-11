import copy
import functools
import json
import os
import random
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import seqio
import tensorflow as tf

import flan.v2.mixtures


##############################################################
##### Instantiate the submixtures with each template style
##############################################################

# ZSOPT, FSOPT, ZSNOOPT, FSNOOPT are template styles.
# ZS means a zero-shot prompt, FS means a few-shot prompt
# OPT means the answer options for tasks with multiple choice answers are included in the template
# NOOPT means the answer options for tasks with multiple choice answers are NOT included in the template

seqio.MixtureRegistry.add(
    'cot_submix',
    tasks=[
        ('cot_zsopt', 1),    # mixing weight = 50%
        ('cot_fsopt', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'dialog_submix',
    tasks=[
        ('dialog_zsopt', 1),    # mixing weight = 50%
        ('dialog_fsopt', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'niv2_submix',
    tasks=[
        ('niv2_zsopt', 1),    # mixing weight = 50%
        ('niv2_fsopt', 1),    # mixing weight = 50%
    ])

seqio.MixtureRegistry.add(
    'flan2021_submix',
    tasks=[
        ('flan_zsopt', 1),      # mixing weight = 25%
        ('flan_fsopt', 1),      # mixing weight = 25%
        ('flan_zsnoopt', 1),    # mixing weight = 25%
        ('flan_fsnoopt', 1),    # mixing weight = 25%
    ])

seqio.MixtureRegistry.add(
    't0_submix',
    tasks=[
        ('t0_zsopt', 1),      # mixing weight = 25%
        ('t0_fsopt', 1),      # mixing weight = 25%
        ('t0_zsnoopt', 1),    # mixing weight = 25%
        ('t0_fsnoopt', 1),    # mixing weight = 25%
    ])

# Define the Final Flan Collection Mixture
seqio.MixtureRegistry.add(
    'flan2022_submix',
    tasks=[
        ('flan2021_submix', 0.4),  # mixing weight = 40%
        ('t0_submix', 0.32),       # mixing weight = 32%
        ('niv2_submix', 0.2),      # mixing weight = 20%
        ('cot_submix', 0.05),      # mixing weight = 5%
        ('dialog_submix', 0.03),   # mixing weight = 3%
    ])

argparser = argparse.ArgumentParser(description='Downloading arguments')

argparser.add_argument('--mixture', type=str, default='cot_zsopt', help='mixture or task to download')
argparser.add_argument("--sample", help='Write only a sample of data to file', action="store_true")

args = argparser.parse_args()


##############################################################
##### See 3 Examples of Mixtures or Submixtures you can try
##############################################################
# 1. Example use cases to use just the chain-of-thought zero-shot data:
print(f'Downloading mixture: {args.mixture}')
selected_mixture = seqio.get_mixture_or_task(args.mixture)

# 2. Example use cases to use just all chain-of-thought templates together:
# selected_mixture = seqio.get_mixture_or_task('cot_submix')

# 3. Example use cases to use the full Flan Collection:
# selected_mixture = seqio.get_mixture_or_task('flan2022_submix')
# This last one (the final Flan Collection mixture) may take too long to run if not
# cached. We suggest starting by caching each of:
# `cot_submix`, `flan2021_submix`, `dialog_submix`, `t0_submix`, `niv2_submix`.

# If you're using Seqio, we suggest caching your mixture as they take a while to generate.
# If you want to read out the post-processed examples into a file, we suggest using the
# sample_fn below to collect 1 epoch of data, according to our mixing rates.
INPUT_SEQ_LEN = 2056
TARGET_SEQ_LEN = 512
dataset = selected_mixture.get_dataset(
    sequence_length={"inputs": INPUT_SEQ_LEN, "targets": TARGET_SEQ_LEN},
    num_epochs=1,
    shuffle=True,
    copy_pretokenized=True,
    # The passthrough features let you track the source/task/template metadata for the example
    passthrough_features=["_template_idx", "_task_source", "_task_name", "_template", "_template_type"]
)

# To read out the data you can do something like this:
source_counter = defaultdict(lambda: 0)
NUM_SAMPLES = 100
# If you would like to take min(1 epoch, NUM_SAMPLES) then use dataset.take(NUM_SAMPLES)
# Or if you would like to gather a full epoch, simply `enumerate(dataset)` until completion.
ds = dataset.take(NUM_SAMPLES) if args.sample else dataset

with open(f"{args.mixture}.jsonl", "w") as f:
    for ex in ds:
        source_counter[ex["_task_source"].numpy()] += 1
        line = {"input" : ex["inputs_pretokenized"].numpy().decode(),
                      "target" : ex["targets_pretokenized"].numpy().decode(),
                      "task_source": ex["_task_source"].numpy().decode(),
                      "template_type" : ex["_template_type"].numpy().decode(),
                      "task_name" : ex["_task_name"].numpy().decode()}
        f.write(json.dumps(line))
        f.write("\n")

print(f"Data Submixture Counts: {source_counter}")
print(line)