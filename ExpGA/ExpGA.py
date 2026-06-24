"""
ExpGA: LIME Explainer-Guided Genetic Algorithm for Fairness Testing (Baseline).

This module implements a fairness testing approach that uses LIME (Local
Interpretable Model-agnostic Explanations) to identify influential features,
then uses a Genetic Algorithm to search for discriminatory inputs.

This serves as the baseline comparison for ChiGA, which replaces LIME with
the simpler and faster Chi-squared feature ranking.

Usage:
    python ExpGA.py --dataset credit --sensitive 13 --max-global 1000 --max-local 500000
"""

import argparse
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

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
global_disc_inputs: set = set()
global_disc_inputs_list: list = []
local_disc_inputs: set = set()
local_disc_inputs_list: list = []
tot_inputs: set = set()

# Track which feature rank the sensitive attribute falls into (diagnostic)
location = np.zeros(21)


# ---------------------------------------------------------------------------
# Dataset & configuration registry
# ---------------------------------------------------------------------------
DATA_REGISTRY = {
    "census": (census_cfg, census_data),
    "credit": (credit_cfg, credit_data),
    "bank": (bank_cfg, bank_data),
    "compas": (compas_cfg, compas_data),
}

# Per-dataset threshold for LIME feature-rank filtering
THRESHOLD_CONFIG = {"census": 7, "credit": 14, "bank": 10, "compas": 7}


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="ExpGA: LIME Explainer-Guided GA for Fairness Testing (Baseline)",
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
        help="Path to the pre-trained fair model (.pkl). "
             "Default: ../ExpGA_fair_models/<dataset>/<sensitive>/MLP_fair1.pkl",
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
        "--output-dir", type=str, default="../ExpGA_results",
        help="Directory to save results (default: ../ExpGA_results).",
    )
    return parser


class GlobalDiscovery:
    """Generates random seed samples for the global discovery phase."""

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


def search_seed(model, feature_names, sens_name, explainer,
                train_vectors, num_features, threshold_l):
    """Use LIME explanations to find seed samples for the GA.

    A sample is selected as a seed if the sensitive attribute ranks
    *above* a threshold in the LIME feature-importance list (i.e., it
    is considered influential by the local explainer).

    Returns at most 200 seeds.
    """
    seeds = []
    for x in train_vectors:
        tot_inputs.add(tuple(x))
        exp = explainer.explain_instance(x, model.predict_proba, num_features=num_features)
        explain_labels = exp.available_labels()
        exp_result = exp.as_list(label=explain_labels[0])

        # Find rank of the sensitive feature
        rank = [item[0] for item in exp_result]
        loc = rank.index(sens_name)
        location[loc] += 1

        if loc < threshold_l:
            seeds.append(x)
        if len(seeds) >= 200:
            return seeds
    return seeds


def evaluate_local(inp, model, config, sensitive_param: int, threshold: float = 0):
    """Fitness function for the Genetic Algorithm.

    Returns
    -------
    (fitness, is_discriminatory) : (float, int)
        fitness is 2 * |pred_diff| + 1 when discriminatory.
    """
    inp_ = [int(v) for v in inp]
    inp_[sensitive_param - 1] = int(config.input_bounds[sensitive_param - 1][0])
    tot_inputs.add(tuple(inp_))

    lo, hi = config.input_bounds[sensitive_param - 1]
    for val in range(lo, hi + 1):
        if val == inp_[sensitive_param - 1]:
            continue

        inp0 = np.reshape(np.asarray(inp_), (1, -1))
        inp1 = np.reshape(
            np.asarray([v if i != sensitive_param - 1 else val
                         for i, v in enumerate(inp_)]),
            (1, -1),
        )

        out0 = model.predict(inp0)
        out1 = model.predict(inp1)

        disc_key = tuple(map(tuple, inp0.tolist()))
        if (abs(out0 - out1) > threshold
                and disc_key not in global_disc_inputs
                and disc_key not in local_disc_inputs):
            local_disc_inputs.add(disc_key)
            local_disc_inputs_list.append(inp0.tolist()[0])
            return 2 * abs(int(out1) - int(out0)) + 1, 1

    return 2 * abs(int(out1) - int(out0)) + 1, 0


def run_expga(dataset: str, sensitive_param: int, model_path: str,
              max_global: int, max_local: int, max_time: int,
              output_dir: str):
    """Run the full ExpGA fairness-testing pipeline (baseline)."""
    config, data_loader = DATA_REGISTRY[dataset]
    feature_names = config.feature_name
    class_names = config.class_name
    sens_name = config.sens_name[sensitive_param]
    params = config.params
    input_bounds = config.input_bounds
    threshold_l = THRESHOLD_CONFIG[dataset]

    print(f"Dataset: {dataset}")
    print(f"Sensitive attribute: {sensitive_param} ({sens_name})")
    print(f"Model: {model_path}")
    print(f"LIME rank threshold: {threshold_l}")
    print(f"Max global samples: {max_global}, Max local iterations: {max_local}")
    print(f"Features ({params}): {feature_names}")
    print(f"Classes: {class_names}")

    model = joblib.load(model_path)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    X, Y, _, _ = data_loader()

    # ------------------------------------------------------------------
    # Global Discovery
    # ------------------------------------------------------------------
    start = time.time()
    model_tag = Path(model_path).name.split("_")[0]

    output_path = Path(output_dir) / dataset / str(sensitive_param)
    output_path.mkdir(parents=True, exist_ok=True)

    log_path = output_path / f"expga_{model_tag}_{dataset}_{sensitive_param}_{max_global // 100}_{max_local // 100}_1H.txt"
    save_path = output_path / f"local_samples_{model_tag}_expga_{max_global}_1H.npy"

    with open(log_path, "a") as f:
        f.write(f"ExpGA run — dataset={dataset} sensitive={sensitive_param} "
                f"max_global={max_global} max_local={max_local}\n\n")

    global_disco = GlobalDiscovery()
    train_samples = np.array(
        global_disco(max_global, params, input_bounds, sensitive_param)
    )
    np.random.shuffle(train_samples)
    print(f"Global discovery: {train_samples.shape[0]} samples generated.")

    # Build LIME explainer
    explainer = LimeTabularExplainer(
        X, feature_names=feature_names, class_names=class_names,
        discretize_continuous=False,
    )
    print("LIME explainer constructed.")

    # Search for seeds where sensitive attribute ranks high in LIME
    print("Searching for seeds via LIME ...")
    seeds = search_seed(model, feature_names, sens_name, explainer,
                        train_samples, params, threshold_l)
    print(f"Found {len(seeds)} seed samples (sensitive rank < {threshold_l}).")

    for inp in seeds:
        inp0 = np.reshape(np.asarray([int(v) for v in inp]), (1, -1))
        global_disc_inputs.add(tuple(map(tuple, inp0)))
        global_disc_inputs_list.append(inp0.tolist()[0])

    print(f"Global discovery: {len(global_disc_inputs_list)} candidates "
          f"(LIME rank < {threshold_l}).")
    print(f"LIME rank distribution: {location[:params]}")

    # ------------------------------------------------------------------
    # Genetic Algorithm Local Search
    # ------------------------------------------------------------------
    print("\nStarting Genetic Algorithm local search ...")
    ga = GA(
        nums=global_disc_inputs_list,
        bound=input_bounds,
        func=lambda x: evaluate_local(x, model, config, sensitive_param),
        DNA_SIZE=len(input_bounds),
        cross_rate=0.9,
        mutation=0.05,
    )

    checkpoint_interval = 300
    next_checkpoint = checkpoint_interval

    for i in range(max_local):
        ga.evolution()
        elapsed = time.time() - start

        if elapsed >= next_checkpoint:
            pct = (len(global_disc_inputs_list) + len(local_disc_inputs_list)) / max(len(tot_inputs), 1) * 100
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

        if elapsed >= max_time:
            pct = (len(global_disc_inputs_list) + len(local_disc_inputs_list)) / max(len(tot_inputs), 1) * 100
            with open(log_path, "a") as f:
                f.write("-------------FINISH------------------\n")
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

    np.save(str(save_path), np.array(local_disc_inputs_list))
    print(f"Results saved to {save_path}")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.model is None:
        args.model = f"../ExpGA_fair_models/{args.dataset}/{args.sensitive}/MLP_fair1.pkl"

    # Clear global state for a fresh run
    global_disc_inputs.clear()
    global_disc_inputs_list.clear()
    local_disc_inputs.clear()
    local_disc_inputs_list.clear()
    tot_inputs.clear()
    location.fill(0)

    run_expga(
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
