import argparse
import concurrent.futures
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from ydc_evals_optimize.metrics.deep_research.deep_research_pairwise_metric import (
    DEFAULT_EVAL_MODEL,
    DeepResearchPairwiseMetric,
    DeepResearchScoreResult,
)


class DeepResearchEvaluator:
    """Evaluator class for evaluating deep reports using the DeepResearchPairwiseMetric."""

    def __init__(
        self,
        model: str = DEFAULT_EVAL_MODEL,
        output_path: Optional[Path] = None,
        evaluator_num_workers: int = 4,
        metric_num_workers: int = 3,
        metric_num_trials: int = 3,
    ):
        """
        Initialize the evaluator.

        Args:
            model: The model to use for evaluation
            output_path: Path to save evaluation results
            evaluator_num_workers: Number of workers for parallel processing of evaluation tasks
            metric_num_workers: Number of workers for the underlying pairwise metric
            metric_num_trials: Number of trials to run for each evaluation
        """
        self.model = model
        self.output_path = output_path
        self.evaluator_num_workers = evaluator_num_workers
        self.metric_num_trials = metric_num_trials

        # Initialize the pairwise evaluator
        self.pairwise_metric = DeepResearchPairwiseMetric(
            eval_model=model,
            num_trials=metric_num_trials,
            num_workers=metric_num_workers,  # Number of workers for the metric
        )

    def evaluate_single(
        self,
        question: str,
        reference_answer: str,
        predicted_answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single question-answer pair.

        Args:
            question: The research question
            reference_answer: The reference answer
            predicted_answer: The predicted answer to evaluate
            metadata: Additional metadata to include in the result

        Returns:
            Dictionary containing evaluation results
        """
        result = {
            "question": question,
            "reference_answer": reference_answer,
            "predicted_answer": predicted_answer,
            "model": self.model,
            "timestamp": time.time(),
        }

        # Include any additional metadata
        if metadata:
            result.update(metadata)

        try:
            # Score the answer using the pairwise evaluator
            score_result = self.pairwise_metric.score(
                question=question,
                reference_answer=reference_answer,
                predicted_answer=predicted_answer,
            )

            # Add scores to the result
            result["success"] = True
            result["score_result"] = score_result.model_dump()

            # Add individual dimension scores for easier analysis
            for dimension in score_result.model_dump():
                dim_data = getattr(score_result, dimension)
                result[f"{dimension}_score"] = dim_data.score
                result[f"{dimension}_grade"] = dim_data.grade
                result[f"{dimension}_is_win"] = dim_data.is_win

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        return result

    def evaluate_batch(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of question-answer pairs.

        Args:
            data: DataFrame containing questions and answers to evaluate
                Expected columns: 'question', 'reference_answer', 'predicted_answer'

        Returns:
            List of evaluation results
        """
        results = []

        # Check if required columns exist
        required_columns = ["question", "reference_answer", "predicted_answer"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Input data must contain column: {col}")

        # Create a list of tasks
        tasks = []
        for _, row in data.iterrows():
            # Extract metadata (any columns not used for evaluation)
            metadata = {
                col: row[col] for col in data.columns if col not in required_columns
            }

            tasks.append(
                {
                    "question": row["question"],
                    "reference_answer": row["reference_answer"],
                    "predicted_answer": row["predicted_answer"],
                    "metadata": metadata,
                }
            )

        # Process tasks in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.evaluator_num_workers
        ) as executor:
            futures = [
                executor.submit(
                    self.evaluate_single,
                    task["question"],
                    task["reference_answer"],
                    task["predicted_answer"],
                    task["metadata"],
                )
                for task in tasks
            ]

            # Process results as they complete
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Evaluating",
            ):
                try:
                    result = future.result()
                    results.append(result)

                except Exception as e:
                    print(f"Error processing task: {e}")

        return results

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary containing aggregated metrics
        """
        # Filter successful evaluations
        successful_results = [r for r in results if r.get("success", False)]

        if not successful_results:
            return {"support": 0, "error": "No successful evaluations found"}

        # Convert results to DeepResearchScoreResult objects
        score_results = []
        for result in successful_results:
            try:
                score_result = DeepResearchScoreResult.model_validate(
                    result["score_result"]
                )
                score_results.append(score_result)
            except Exception as e:
                print(f"Error parsing score result: {e}")

        # Use the evaluator's aggregate method to get aggregate metrics
        return self.pairwise_metric.aggregate(score_results)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Deep Research pairwise evaluations"
    )
    parser.add_argument(
        "--input-data",
        type=str,
        required=True,
        help="Path to input CSV file. The CSV should have columns 'question', 'reference_answer', and 'predicted_answer'.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save results"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--evaluator-num-workers",
        type=int,
        default=4,
        help="Number of worker threads for evaluator",
    )
    parser.add_argument(
        "--metric-num-workers",
        type=int,
        default=3,
        help="Number of worker threads for metric",
    )
    parser.add_argument(
        "--metric-num-trials",
        type=int,
        default=3,
        help="Number of trials per metric computation. The higher this number, the more stable the metric will be, but it will also take longer to compute.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up output path
    output_path = output_dir / f"deep_research_results_{args.model}.jsonl"

    # Load input data
    print(f"Loading data from {args.input_data}")
    df = pd.read_csv(args.input_data)
    print(f"Loaded {len(df)} examples")

    # Initialize evaluator
    evaluator = DeepResearchEvaluator(
        model=args.model,
        output_path=output_path,
        evaluator_num_workers=args.evaluator_num_workers,
        metric_num_workers=args.metric_num_workers,
        metric_num_trials=args.metric_num_trials,
    )

    # Run evaluation
    print(
        f"Starting evaluation with model {args.model} using {args.evaluator_num_workers} evaluator workers and {args.metric_num_workers} metric workers..."
    )
    results = evaluator.evaluate_batch(df)

    print(f"Results saved to {output_path}")
    pd.DataFrame(results).to_json(output_path, orient="records", lines=True)

    # Load and display summary
    results_df = pd.read_json(output_path, lines=True)
    print("\nResults summary:")
    print(f"Model: {args.model}")
    print(f"Total evaluations: {len(results_df)}")
    print(
        f"Successful evaluations: {len(results_df[results_df.get('success', False)])}"
    )
    print(f"Failed evaluations: {len(results_df[~results_df.get('success', False)])}")

    # Compute aggregate metrics
    if len(results) > 0:
        print("Aggregating results...")
        aggregate_metrics = evaluator.aggregate_results(results)

        # Save aggregate metrics
        aggregate_path = output_dir / f"deep_research_aggregate_{args.model}.json"
        with open(aggregate_path, "w") as f:
            json.dump(aggregate_metrics, f, indent=2)

        print(f"Aggregate metrics saved to {aggregate_path}")

        # Display key metrics
        print("\nKey Metrics:")
        print(f"Total examples: {aggregate_metrics.get('support', 0)}")

        if "macro_avg" in aggregate_metrics:
            print("\nMacro Average Metrics:")
            for metric, value in aggregate_metrics["macro_avg"].items():
                if isinstance(value, float):
                    print(f"{metric}: {value:.4f}")
                else:
                    print(f"{metric}: {value}")


if __name__ == "__main__":
    main()
