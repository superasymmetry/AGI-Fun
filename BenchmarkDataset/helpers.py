from iterators import define_iterator, classify_iterator, generate_iterator, edit_iterator
import os
import json
import shutil
import pandas as pd
from collections import Counter


from iterators import (
    define_iterator,
    classify_iterator,
    generate_iterator,
    edit_iterator,
)

def count_inferences():
    iter_funcs = {
        'define'  : define_iterator,
        'classify': classify_iterator,
        'generate': generate_iterator,
        'edit'    : edit_iterator,
    }

    counts = {}
    define_model_counts = Counter()

    # 1) Count define *and* build model breakdown
    # -------------------------------------------------
    # We pull define_iterator once, so we can both count total
    # and increment per-model.
    define_iter = define_iterator()
    for meta, _ in define_iter:
        counts.setdefault('define', 0)
        counts['define'] += 1
        define_model_counts[meta['Model']] += 1

    # 2) Count the other tasks
    # -------------------------------------------------
    for name, make_iter in iter_funcs.items():
        if name == 'define':
            continue
        iterator = make_iter()
        counts[name] = sum(1 for _ in iterator)

    # 3) Print totals and percentages
    # -------------------------------------------------
    total = sum(counts.values()) or 1  # avoid div zero

    print("\nOverall inference counts:")
    for name, cnt in counts.items():
        pct = cnt / total * 100
        print(f"{name:>8} → {cnt:5d} inferences ({pct:6.2f}%)")

    # 4) Print define-breakdown by model
    # -------------------------------------------------
    define_total = counts.get('define', 0) or 1
    print("\n‘define’ breakdown by model:")
    for model, cnt in define_model_counts.most_common():
        pct = cnt / define_total * 100
        print(f"  {model:>12} → {cnt:4d} ({pct:5.2f}%)")


def edit_model_breakdown():
    """
    Counts and prints the number (and share) of inferences per Model
    for the Edit task.
    """
    counts = Counter()

    # 1) Tally up all edit inferences by model
    for meta, _ in edit_iterator():
        model = meta.get('Model', 'UNKNOWN')
        counts[model] += 1

    total = sum(counts.values()) or 1  # guard against zero

    # 2) Print sorted breakdown
    print("\n‘edit’ task — inferences by model:")
    for model, cnt in counts.most_common():
        pct = cnt / total * 100
        print(f"  {model:>12} → {cnt:4d} ({pct:5.2f}%)")

    return counts  # in case you want to use it programmatically


def edit_domain_breakdown():
    """
    Counts and prints the number (and share) of inferences per Domain
    for the Edit task.
    """
    counts = Counter()

    # Tally up all edit inferences by domain
    for meta, _ in edit_iterator():
        domain = meta.get('Domain', 'Unknown')
        counts[domain] += 1

    total = sum(counts.values()) or 1  # guard against zero

    # Print sorted breakdown
    print("\n‘edit’ task — inferences by domain:")
    for domain, cnt in counts.most_common():
        pct = cnt / total * 100
        print(f"  {domain:>20} → {cnt:4d} ({pct:5.2f}%)")

    return counts  # return for programmatic use


def classify_domain_breakdown():
    """
    Counts and prints the number (and share) of inferences per Domain
    for the Classify task, and also lists all Concepts and Models seen.
    """
    counts = Counter()
    concepts = set()
    models = set()

    # Tally up all classify inferences by domain, and collect concepts & models
    for meta, _ in classify_iterator():
        domain = meta.get('Domain', 'Unknown')
        counts[domain] += 1

        concept = meta.get('Concept')
        if concept is not None:
            concepts.add(concept)

        model = meta.get('Model')
        if model is not None:
            models.add(model)

    total = sum(counts.values()) or 1  # guard against zero

    # Print sorted breakdown
    print("\n‘classify’ task — inferences by domain:")
    for domain, cnt in counts.most_common():
        pct = cnt / total * 100
        print(f"  {domain:>20} → {cnt:4d} ({pct:5.2f}%)")

    # Print all concepts encountered
    print("\nConcepts seen:")
    for c in sorted(concepts):
        print(f"  - {c}")

    # Print all models encountered
    print("\nModels seen:")
    for m in sorted(models):
        print(f"  - {m}")

def classify_concept_breakdown(domain_filter: str = None):
    """
    Counts and prints the number (and share) of inferences per Concept
    for the Classify task. If `domain_filter` is provided, only inferences
    in that Domain are counted.

    Parameters
    ----------
    domain_filter : str, optional
        If set (e.g. "Literary Techniques"), only inferences whose
        meta['Domain'] equals this string will be included.
    """
    counts = Counter()

    # Tally up classify inferences by Concept (optionally filtering by Domain)
    for meta, _ in classify_iterator():
        domain = meta.get('Domain', 'Unknown')
        if domain_filter is None or domain == domain_filter:
            concept = meta.get('Concept', 'Unknown')
            counts[concept] += 1

    total = sum(counts.values()) or 1  # guard against division by zero

    # Header
    header = f"\n‘classify’ task — inferences by Concept"
    if domain_filter:
        header += f" (Domain: {domain_filter})"
    print(header)

    # Print sorted breakdown
    for concept, cnt in counts.most_common():
        pct = cnt / total * 100
        print(f"  {concept:>30} → {cnt:4d} ({pct:5.2f}%)")

if __name__ == '__main__':
    # count_inferences()
    # edit_model_breakdown()
    # edit_domain_breakdown()
    classify_domain_breakdown()
    # classify_concept_breakdown()