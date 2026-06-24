"""
ChiGA: Chi-squared Guided Genetic Algorithm for Fairness Testing.

This module implements a fairness testing approach that uses the Chi-squared
statistical test to rank feature importance, then guides a Genetic Algorithm
to efficiently discover discriminatory inputs without relying on local
explanation methods (e.g., LIME or SHAP).

Workflow:
    1. Global Discovery: randomly sample the input space.
    2. Chi-squared feature ranking on training data.
    3. Genetic Algorithm local search with chi-squared-guided mutation.
    4. Output discriminatory samples found.

Usage:
    python ChiGA.py --dataset credit --sensitive 13 --max-global 1000 --max-local 500000
"""

import argparse
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

from Genetic_Algorithm import GA
from data.bank import bank_data
from data.census import census_data
from data.compas import compas_data
from data.credit import credit_data
from utils.config import bank as bank_cfg
from utils.config import census as census_cfg
from utils.config import compas as compas_cfg
from utils.config import credit as credit_cfg

# ---------------------------------------------------------------------------
# Global state (shared across the GA population for deduplication)
# ---------------------------------------------------------------------------
global_inputs: set = set()
global_inputs_list: list = []
local_disc_inputs: set = set()
local_disc_inputs_list: list = []
tot_inputs: set = set()


# ---------------------------------------------------------------------------
# Dataset & configuration registry
# ---------------------------------------------------------------------------
DATA_REGISTRY = {
    "census": (census_cfg, census_data),
    "credit": (credit_cfg, credit_data),
    "bank": (bank_cfg, bank_data),
    "compas": (compas_cfg, compas_data),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="ChiGA: Chi-squared Guided Genetic Algorithm for Fairness Testing",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["census", "credit", "bank", "compas"],
        help="Dataset to run fairness testing on.",
    )
    parser.add_argument(
        "--sensitive", type=int, required=True,
        help="Index of the sensitive attribute (see config.py for per-dataset values).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to the pre-trained model (.pkl). "
             "Default: ../unfair_models/<dataset>/MLP_unfair1.pkl",
    )
    parser.add_argument(
        "--max-global", type=int, default=1000,
        help="Number of random samples for global discovery (default: 1000).",
    )
    parser.add_argument(
        "--max-local", type=int, default=500000,
        help="Maximum GA iterations for local search (default: 500000).",
    )
    parser.add_argument(
        "--max-time", type=int, default=3600,
        help="Maximum runtime in seconds (default: 3600).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="../results",
        help="Directory to save results (default: ../results).",
    )
    return parser


class GlobalDiscovery:
    """Generates random seed samples for the global discovery phase.

    Samples are drawn uniformly from each feature's valid range.
    The sensitive attribute is fixed to its minimum value.
    """

    def __init__(self, stepsize: int = 1):
        self.stepsize = stepsize

    def __call__(self, n_samples: int, n_params: int,
                 bounds: list, sensitive_idx: int) -> list:
        samples = []
        random.seed(time.time())
        while len(samples) < n_samples:
            x = np.array([
                random.randint(bounds[i][0], bounds[i][1])
                for i in range(n_params)
            ])
            x[sensitive_idx - 1] = bounds[sensitive_idx - 1][0]
            samples.append(x)
        return samples


def evaluate_local(inp, model, config, sensitive_param: int):
    """Fitness function for the Genetic Algorithm.

    A sample is *discriminatory* if varying only the sensitive attribute
    produces different model predictions.  The fitness score is proportional
    to the prediction probability spread across sensitive-attribute values.

    Returns
    -------
    (fitness, is_discriminatory) : (float, int)
    """
    inp0 = [int(v) for v in inp]
    inp0[sensitive_param - 1] = int(config.input_bounds[sensitive_param - 1][0])
    inp0_arr = np.reshape(inp0, (1, -1))

    tot_inputs.add(tuple(map(tuple, inp0_arr)))

    min_pre = 1.0
    max_pre = 0.0
    sum0 = 0
    sum1 = 0

    lo, hi = config.input_bounds[sensitive_param - 1]
    for val in range(lo, hi + 1):
        inp1 = [int(v) for v in inp]
        inp1[sensitive_param - 1] = val
        inp1_arr = np.reshape(np.asarray(inp1), (1, -1))

        pred = model.predict(inp1_arr)
        if pred == 1:
            sum1 += 1
        else:
            sum0 += 1

        proba = model.predict_proba(inp1_arr)
        if min_pre > proba[0, 1]:
            min_pre = proba[0, 1]
        if max_pre < proba[0, 1]:
            max_pre = proba[0, 1]

    disc_key = tuple(map(tuple, list(inp0_arr)))
    if (sum0 != 0) and (sum1 != 0) and (disc_key not in local_disc_inputs):
        local_disc_inputs.add(disc_key)
        local_disc_inputs_list.append(inp0_arr.tolist()[0])
        return abs(max_pre - min_pre) * 5, 1

    return abs(max_pre - min_pre), 0


def run_chiga(dataset: str, sensitive_param: int, model_path: str,
              max_global: int, max_local: int, max_time: int,
              output_dir: str):
    """Run the full ChiGA fairness-testing pipeline.

    Parameters
    ----------
    dataset : str
        One of {"census", "credit", "bank", "compas"}.
    sensitive_param : int
        1-based index of the sensitive attribute.
    model_path : str
        Path to a pickled sklearn-compatible classifier.
    max_global : int
        Number of random seeds for the global discovery phase.
    max_local : int
        Maximum GA iterations for the local search phase.
    max_time : int
        Wall-clock time budget in seconds.
    output_dir : str
        Directory where results (.npy and log) are written.
    """
    config, data_loader = DATA_REGISTRY[dataset]
    feature_names = config.feature_name
    class_names = config.class_name
    sens_name = config.sens_name[sensitive_param]
    params = config.params
    input_bounds = config.input_bounds

    print(f"Dataset: {dataset}")
    print(f"Sensitive attribute: {sensitive_param} ({sens_name})")
    print(f"Model: {model_path}")
    print(f"Max global samples: {max_global}, Max local iterations: {max_local}")
    print(f"Features ({params}): {feature_names}")
    print(f"Classes: {class_names}")

    model = joblib.load(model_path)

    # ------------------------------------------------------------------
    # Load data & compute Chi-squared feature importance
    # ------------------------------------------------------------------
    X, Y, _, _ = data_loader()

    # Shift negative-valued features to be non-negative (required by chi2)
    for b in range(len(input_bounds)):
        if input_bounds[b][0] < 0:
            X[:, b] -= input_bounds[b][0]

    # Convert one-hot labels to 1-D
    Y_1d = np.array([y[1] for y in Y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y_1d, test_size=0.2, random_state=1,
    )

    chi_scores, _ = chi2(X_train, y_train)
    chi_scores = np.array(chi_scores)

    prob_positive = chi_scores / np.sum(chi_scores)
    prob_negative = (1.0 / prob_positive) / np.sum(1.0 / prob_positive)

    print(f"Chi-squared positive weights: {prob_positive}")
    print(f"Chi-squared negative weights: {prob_negative}")

    # ------------------------------------------------------------------
    # Global Discovery
    # ------------------------------------------------------------------
    start = time.time()
    model_tag = Path(model_path).name.split("_")[0]

    output_path = Path(output_dir) / dataset / str(sensitive_param)
    output_path.mkdir(parents=True, exist_ok=True)

    log_path = output_path / f"chiga_{model_tag}_{dataset}_{sensitive_param}_{max_global // 100}_{max_local // 100}_1H.txt"
    save_path = output_path / f"disc_samples_{model_tag}_chiga_{max_global}_{max_local // 100}_1H.npy"

    with open(log_path, "a") as f:
        f.write(f"ChiGA run — dataset={dataset} sensitive={sensitive_param} "
                f"max_global={max_global} max_local={max_local}\n\n")

    global_disco = GlobalDiscovery()
    train_samples = np.array(
        global_disco(max_global, params, input_bounds, sensitive_param)
    )
    np.random.shuffle(train_samples)
    print(f"Global discovery: {train_samples.shape[0]} samples generated.")

    for inp in train_samples:
        inp0 = np.reshape(np.asarray([int(v) for v in inp]), (1, -1))
        tot_inputs.add(tuple(map(tuple, inp0)))
        global_inputs.add(tuple(map(tuple, inp0)))
        global_inputs_list.append(inp0.tolist()[0])

    print(f"Global discovery completed in {time.time() - start:.1f}s")

    # ------------------------------------------------------------------
    # Genetic Algorithm Local Search
    # ------------------------------------------------------------------
    print("\nStarting Genetic Algorithm local search ...")
    ga = GA(
        nums=global_inputs_list,
        bound=input_bounds,
        func=lambda x: evaluate_local(x, model, config, sensitive_param),
        sensitive_param=sensitive_param,
        datasetName=dataset,
        DNA_SIZE=len(input_bounds),
        cross_rate=0.9,
        mutation=0.05,
        mutation_positive=prob_positive,
        mutation_negative=prob_negative,
    )

    checkpoint_interval = 300  # seconds between log writes
    next_checkpoint = checkpoint_interval

    for i in range(max_local):
        ga.evolution()
        elapsed = time.time() - start

        # Periodic logging
        if elapsed >= next_checkpoint:
            pct = len(local_disc_inputs_list) / max(len(tot_inputs), 1) * 100
            with open(log_path, "a") as f:
                f.write(f"Percentage discriminatory inputs - {pct:.4f}\n")
                f.write(f"Number of discriminatory inputs are {len(local_disc_inputs_list)}\n")
                f.write(f"Total Inputs are {len(tot_inputs)}\n")
                f.write(f"Use time: {elapsed:.1f}s\n\n")
            print(f"[{elapsed:.0f}s] Discriminatory: {len(local_disc_inputs_list)} "
                  f"/ {len(tot_inputs)} ({pct:.2f}%)")
            next_checkpoint += checkpoint_interval

        if i % 300 == 0:
            pct = len(local_disc_inputs_list) / max(len(tot_inputs), 1) * 100
            print(f"Epoch {i}: {len(local_disc_inputs_list)} discriminatory "
                  f"/ {len(tot_inputs)} total ({pct:.2f}%) — {elapsed:.0f}s")

        # Time budget exhausted
        if elapsed >= max_time:
            pct = len(local_disc_inputs_list) / max(len(tot_inputs), 1) * 100
            with open(log_path, "a") as f:
                f.write("------------------FINISH---------------------\n")
                f.write(f"Percentage discriminatory inputs - {pct:.4f}\n")
                f.write(f"Number of discriminatory inputs are {len(local_disc_inputs_list)}\n")
                f.write(f"Total Inputs are {len(tot_inputs)}\n")
                f.write(f"Use time: {elapsed:.1f}s\n")
                f.write(f"Saved to: {save_path}\n")
            np.save(str(save_path), np.array(local_disc_inputs_list))
            print(f"\nFinished! Results saved to {save_path}")
            print(f"  {len(local_disc_inputs_list)} discriminatory samples found "
                  f"out of {len(tot_inputs)} total ({pct:.2f}%)")
            return

    # If loop completes without hitting time budget
    np.save(str(save_path), np.array(local_disc_inputs_list))
    print(f"Results saved to {save_path}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.model is None:
        args.model = f"../unfair_models/{args.dataset}/MLP_unfair1.pkl"

    # Clear global state for a fresh run
    global_inputs.clear()
    global_inputs_list.clear()
    local_disc_inputs.clear()
    local_disc_inputs_list.clear()
    tot_inputs.clear()

    run_chiga(
        dataset=args.dataset,
        sensitive_param=args.sensitive,
        model_path=args.model,
        max_global=args.max_global,
        max_local=args.max_local,
        max_time=args.max_time,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
