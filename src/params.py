# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
from pathlib import Path

def get_project_root():
    return Path(__file__).parent.parent

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    if model_name in ["RN50", "RN101", "RN50x4", "RN50x64",  "RN50x16", "RN50_flat", "RN50_t1", "RN50_t2", "RN50_t3", "RN50_t4", "RN50_t5", "RN50_t6",
                      "RN50_flat_ft", "RN50_t1_pos_ft", "RN50_t2_pos_ft", "RN50_t1_pos", "RN50_t2_pos",
                      "RN50_flat_large", "RN50_t1_large", "RN50_t2_large",
                      "RN50_a2", "RN50_a2s", "ViT-H-14"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}
    elif model_name in ["ViT-B/32", "ViT-L/14", "ViT-B/16"]:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-time-suffix",
        default=True,
        action="store_false",
        help="Whether to append current time in the suffix.",
        dest="time_suffix")
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to csv filewith training data",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to csv file with validation data",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=None,
        help="list of prompts split with ,",
    )
    parser.add_argument(
        "--retrieval-data",
        type=str,
        default=None,
        help="Path to csv file or folder of retrieval data",
    )
    parser.add_argument(
        "--demo-out",
        type=str,
        default="demo",
        help="Path to the output directory for visualization",
    )
    parser.add_argument(
        "--source-data",
        type=str,
        default=None,
        help="Path to txt file of retrieval data",
    )
    parser.add_argument(
        "--target-data",
        type=str,
        default=None,
        help="Path to txt file of retrieval data",
    )
    parser.add_argument(
        "--target-pad",
        action="store_true",
        default=False,
        help="Padding augmentation proposed by combiner.",
    )
    parser.add_argument(
        "--query_file",
        type=str,
        default=None,
        help="Path to query image file for retrieval visualization",
    )
    parser.add_argument("--eval-mode",
        type=str,
        choices=["coco", "cirr", "cirr_test", "fashion", "imgnet"],
        default="coco",
        help="Evaluate Pacs")
    parser.add_argument("--middle_dim",
        default=512,
        type=int,
        help="Number of hidden units in mapping network.")
    parser.add_argument("--droprate",
        default=0.1,
        type=float,
        help="Dropout rate.")
    parser.add_argument(
        "--n-layer", type=int, default=2, help="Number of layers in im2text"
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "inet", "auto", "inet,csv", "csv,inet", "directory", "fashion-iq", "cirr", "imgnet_r"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-type-val",
        choices=["webdataset", "csv", "inet", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument("--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument("--use-debiased-sampler",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument("--use-prefix",
        default=False,
        action="store_true",
        help="Whether to use prefix conditioning in using image classification dataset.")
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Specify a single GPU to run the code on for debugging."
        "Leave at None to use all available GPUs.",
    )
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--regression-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "fp16", "fp32"],
        default="amp",
        help="Floating point precition."
    )
    parser.add_argument(
        "--model",
        choices=["RN50", "RN101", "RN50x4",  "RN50x64", "RN50x16", "ViT-B/16", "ViT-B/32", "ViT-L/14", "ViT-H-14", 
                 "RN50_flat", "RN50_t1", "RN50_t2", "RN50_t3", "RN50_t4", "RN50_t5", "RN50_t6",
                 "RN50_flat_ft", "RN50_t1_pos_ft", "RN50_t2_pos_ft", "RN50_t1_pos", "RN50_t2_pos",
                 "RN50_flat_large", "RN50_t1_large", "RN50_t2_large",
                 "RN50_a2", "RN50_a2s"],
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--openai-pretrained",
        default=False,
        action='store_true',
        help="Use the openai pretrained models.",
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:6100",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--skip-aggregate",
        default=False,
        action="store_true",
        help="whether to aggregate features across gpus before computing the loss"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--C", type=float, default=3.16, help="inverse regularizer for logistic reg."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log diretory, and execute from there."
    )
    parser.add_argument(
        "--dp",
        default=False,
        action="store_true",
        help="Use DP instead of DDP."
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="In DP, which GPUs to use for multigpu training",
    )
    args = parser.parse_args()
    args.aggregate = not args.skip_aggregate

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args
