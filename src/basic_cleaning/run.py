#!/usr/bin/env python
"""
Download the data from W&B, do the cleaning task on it and store the cleaned data as an artifact in W&B.
"""
import os
import argparse
import logging
from typing import Any
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args:Any):
    """_summary_

    Args:
        args (_type_): _description_
    """
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    # Performing the cleaning
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Storing the Artifact in W&B
    df.to_csv(args.output_artifact, index=False)
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

    os.remove(args.output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic cleaning")

    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="The artifact type.",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="The object we want to store in the W&B.",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the object we want to store.",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="The description of the file that we are saving in the W&B.",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=int,
        help="The minimum accepted price for the houses.",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=int,
        help="The maximum accepted price for the houses.",
        required=True
    )


    args = parser.parse_args()

    go(args)
