#!/usr/bin/env python3
"""Given a data file with LM QA predictions, evaluate the predictions.
"""
import argparse
import json
import logging
import statistics
import sys
from copy import deepcopy

from tqdm import tqdm
from xopen import xopen

from .metrics import best_subspan_em

logger = logging.getLogger(__name__)

METRICS = [
    (best_subspan_em, "best_subspan_em"),
]


def main(
    input_path,
    output_path,
    logger
):
    all_examples = []
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            all_examples.append(input_example)

    # Compute normal metrics in parallel, if applicable
    logger.info("Computing metrics")
    all_example_metrics = []
    for example in tqdm(all_examples):
        all_example_metrics.append(get_metrics_for_example(example))

    # Average metrics across examples
    for (_, metric_name) in METRICS:
        average_metric_value = statistics.mean(
            example_metrics[metric_name] for (example_metrics, _) in all_example_metrics
        )
        print(f"{metric_name}: {average_metric_value}")
        logger.info(f"{metric_name}: {average_metric_value}")
    return average_metric_value

    # if output_path:
    #     with xopen(output_path, "w") as f:
    #         for (example_metrics, example) in all_example_metrics:
    #             example_with_metrics = deepcopy(example)
    #             for metric_name, metric_value in example_metrics.items():
    #                 example_with_metrics[f"metric_{metric_name}"] = metric_value
    #             f.write(json.dumps(example_with_metrics) + "\n")


def get_metrics_for_example(example):
    gold_answers = example["answers"]
    model_answer = example["model_answer"]

    # NOTE: we take everything up to the first newline, since otherwise models could hack
    # the metric by simply copying te input context (as the gold answer is guaranteed
    # to occur in the input context).
    model_answer = model_answer.split("\n")[0].strip()

    example_metrics = {}
    for (metric, metric_name) in METRICS:
        example_metrics[metric_name] = metric(prediction=model_answer, ground_truths=gold_answers)
    return (example_metrics, example)

def evaluate_qa(input_path,output_path,logger):
    return main(input_path,output_path,logger)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO, filename='/data/wzh/paperproject/Ms/Ms-PoE/utils/lost_in_the_middle/result_test.log', filemode='a')
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with model predictions and answers.", required=True)
    parser.add_argument(
        "--output-path",
        help="Path to write data with model predictions, answers, and scores.",
    )
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.output_path,
        logger
    )
    logger.info("finished running %s", sys.argv[0])