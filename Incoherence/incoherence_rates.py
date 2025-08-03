import pandas as pd
import math

def compute_accuracy_and_se(count_correct: int, n: int):
    """
    Given number of correct predictions and total n, return (proportion_correct, standard_error)
    """
    if n == 0:
        return 0.0, 0.0
    p = count_correct / n
    se = math.sqrt(p * (1 - p) / n)
    return p, se

def print_incoherence_by_model(csv_path: str = "./inferences/coherence_results.csv"):
    """
    Incoherence rate = 2*(1 − accuracy), expressed as a percentage.
    Prints ±2×SE with raw counts in their own aligned column, for each model and one Overall row.
    """
    # 1) Load & filter out unwanted rows
    df = pd.read_csv(csv_path)
    df = df[df["Concept"].astype(str).str.strip() != "Demanding Bias"]
    df = df[df["Model"].astype(str).str.strip() != "mistralai/Mistral-7B-Instruct-v0.2"]

    # 2) Normalize 'Correct'
    df["Correct"] = (
        df["Correct"]
          .astype(str)
          .str.strip()
          .str.lower()
    )

    # 3) Prepare formatting widths
    name_w, value_w, count_w = 45, 34, 20

    # 4) Print header
    print("\nIncoherence rate by model (2*(1 − accuracy)):\n")
    print(f"{'Model':<{name_w}}{'Incoherence Rate ± SE':<{value_w}}{'Counts':>{count_w}}")
    print("-" * (name_w + value_w + count_w))

    # 5) Per‐model stats
    total_all = 0
    correct_all = 0
    for model in sorted(df["Model"].unique()):
        sub = df[df["Model"] == model]
        n = len(sub)
        c = (sub["Correct"] == "yes").sum()
        total_all += n
        correct_all += c

        p, se = compute_accuracy_and_se(c, n)
        incoh  = 2 * (1 - p) * 100
        margin = 2 * se * 100

        value_str  = f"{incoh:6.2f}%±{margin:4.1f}%"
        counts_str = f"{c}/{n}"
        print(f"{model:<{name_w}}{value_str:<{value_w}}{counts_str:>{count_w}}")

    # 6) Overall row
    print("-" * (name_w + value_w + count_w))
    if total_all > 0:
        p_all, se_all = compute_accuracy_and_se(correct_all, total_all)
        incoh_all  = 2 * (1 - p_all) * 100
        margin_all = 2 * se_all * 100

        name_str   = "Overall"
        value_str  = f"{incoh_all:6.2f}%±{margin_all:4.1f}%"
        counts_str = f"{correct_all}/{total_all}"
    else:
        name_str, value_str, counts_str = "Overall", "N/A", ""

    print(f"{name_str:<{name_w}}{value_str:<{value_w}}{counts_str:>{count_w}}\n")
