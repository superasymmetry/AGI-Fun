import os
import json
import shutil
import pandas as pd
from constants import literature, psychological_biases, game_theory, models_to_short_name


ALLOWED_MODELS = {
    "Llama-3.3",
    "GPT-4o",
    "Claude-Sonnet",
    "Gemini-2.0",
    "DeepSeek-V3",
    "DeepSeek-R1",
    "Qwen2-VL"
}

ALLOWED_CONCEPTS = {"Haiku", "Shakespearean Sonnet", "Analogy", "Paradox", "Anacoluthon", "Asyndeton", "Hyperbaton", "Synesis", "Accismus", "Slant Rhyme", "Enthymeme", "Anapest", "Fundamental Attribution Error", "Black and White Thinking", "Sunk Cost Fallacy", "IKEA Effect", "Pseudocertainty Effect", "Endowment Effect", "Naive Cynicism", "Normalcy Bias", "Spotlight Effect", "Illusory Superiority", "Catastrophizing", "Strict Dominance", "Iterated Dominance", "Weak Dominance", "Pure Nash Equilibrium", "Mixed Strategy Nash Equilibrium", "Pareto Optimality", "Best Response", "Zero-Sum Game", "Symmetric Game"}


def _get_domain(concept):
    if concept in psychological_biases:
        return 'Psychological biases'
    if concept in game_theory:
        return 'Game theory'
    if concept in literature:
        return 'Literary techniques'
    return 'Unknown'

# Iterator for the define task
def define_iterator(allowed_models=None):
    if allowed_models is None:
        allowed_models = ALLOWED_MODELS

    csv_path = './define/define_labels.csv'
    inference_root = './define/inferences'

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Concept', 'Model'])

    # normalize whitespace and filter
    df['Model'] = df['Model'].astype(str).str.strip()
    df = df[df['Model'].isin(allowed_models)]
    df = df[df['Concept'].isin(ALLOWED_CONCEPTS)]

    for _, row in df.iterrows():
        concept = str(row['Concept']).strip()
        model = str(row['Model']).strip()
        filename = str(row['File']).strip()

        inference_path = os.path.join(inference_root, concept, model, filename)

        inference_content = None
        if os.path.isfile(inference_path):
            with open(inference_path, 'r', encoding='utf-8') as f:
                inference_content = f.read()
        else:
            pass

        yield row.to_dict(), inference_content

# Iterator for the classify task
def classify_iterator(
    psych_csv: str = './classify/psych_classify_with_cot.csv',
    other_csv: str = './classify/literature_and_game_theory_classify_with_cot.csv'
):
    """
    Yields (metadata_dict, inference_content) for each classified example, with metadata
    normalized to keys: Concept, Correct, Domain, File, Model, Task.
    - Correct is 'yes' if the original Correct == 1.0, else 'no'.
    - Domain is one of 'Psychological Biases', 'Game Theory', or 'Literary Techniques'.
    - Model is the short name from models_to_short_name.
    - Task is always 'Classify'.
    """
    # helper to map concept → domain
    def get_domain(concept):
        if concept in psychological_biases:
            return 'Psychological biases'
        if concept in game_theory:
            return 'Game theory'
        if concept in literature:
            return 'Literary techniques'
        return 'Unknown'

    # load and concat both CSVs
    dfs = []
    for path in (psych_csv, other_csv):
        df = pd.read_csv(path)
        df = df.dropna(subset=['Concept', 'Model', 'Inference', 'Correct'])
        dfs.append(df)
    df_all = pd.concat(dfs, ignore_index=True)

    for _, row in df_all.iterrows():
        concept = str(row['Concept']).strip()
        # normalize correctness
        correct_flag = 'yes' if float(row['Correct']) == 1.0 else 'no'
        # domain
        domain = get_domain(concept)
        if domain == 'Unknown':
            continue

        # short model name
        full_model = str(row['Model']).strip()
        model = models_to_short_name.get(full_model, full_model)
        # filename
        filename = "psych_classify_with_cot.csv" if domain == 'Psychological Biases' else "literature_and_game_theory_classify_with_cot.csv"

        # build metadata dict
        meta = {
            'Concept': concept,
            'Correct': correct_flag,
            'Domain': domain,
            'File': filename,
            'Model': model,
            'Task': 'Classify'
        }

        # inference content
        inference_content = str(row['Inference']).strip()

        yield meta, inference_content

def generate_iterator(
    csv_path: str = './generate/author_labels_generate.csv',
    root_dir: str = './generate'
):
    """
    Yields (record_dict, content) for every generated example.
    Logs and skips any CSV row whose inference folder or file is missing.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Concept', 'Model', 'File', 'Correct'])

    # 1) Game‐theory branch (unchanged)
    for concept in game_theory:
        for model_short in models_to_short_name.values():
            model_dir = os.path.join(root_dir, 'inferences', concept, model_short)
            if not os.path.isdir(model_dir):
                continue

            for filename in os.listdir(model_dir):
                src = os.path.join(model_dir, filename)
                if not os.path.isfile(src):
                    continue

                with open(src, 'r', encoding='utf-8') as f:
                    raw = f.read()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"[GameTheory] JSON decode error in {src}, skipping")
                    continue

                corr = data.get('correct')
                correct_flag = 'yes' if (corr is True or (isinstance(corr, list) and corr and corr[0] is True)) else 'no'

                rec = {
                    'Concept': concept,
                    'Correct': correct_flag,
                    'Domain': _get_domain(concept),
                    'File': filename,
                    'Model': model_short,
                    'Task': 'Generate'
                }
                for k, v in data.items():
                    if k in ('concept', 'correct'):
                        continue
                    rec[k] = v

                content = data.get('inferences')
                yield rec, content

    # 2) Literature & psychological‐biases branch
    for idx, row in df.iterrows():
        concept     = str(row['Concept']).strip()
        if concept in game_theory:
            continue

        filename    = str(row['File']).strip()
        full_model  = str(row['Model']).strip()
        model_short = models_to_short_name.get(full_model, full_model)
        correct_flag= 'yes' if str(row['Correct']).strip().lower() in ('yes','1','true') else 'no'

        rec = row.to_dict()
        rec.update({
            'Correct': correct_flag,
            'Domain':  _get_domain(concept),
            'File':    filename,
            'Model':   model_short,
            'Task':    'Generate'
        })

        # check that the model directory exists
        model_dir = os.path.join(root_dir, 'inferences', concept, model_short)
        if not os.path.isdir(model_dir):
            # print(f"[MissingDir] Concept={concept!r}, Model={model_short!r} – no directory at {model_dir}, skipping row {idx}")
            continue

        # check that the inference file exists
        inf_path = os.path.join(model_dir, filename)
        if not os.path.isfile(inf_path):
            # print(f"[MissingFile] Concept={concept!r}, Model={model_short!r}, File={filename!r} – file not found, skipping row {idx}")
            continue

        # load and extract
        try:
            with open(inf_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content = data.get('inferences')
        except json.JSONDecodeError:
            print(f"[JSONError] Could not parse JSON in {inf_path}, skipping row {idx}")
            continue

        yield rec, content


def edit_iterator(
    csv_path: str = './edit/author_labels_edit.csv',
    root_dir: str = './edit'
):
    """
    Yields (record_dict, content) for every edit example.
    Logs and skips any entries whose inference JSON is missing or invalid.
    """
    # load CSV for non-game-theory
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['Concept', 'Model', 'File', 'Correct'])

    # 1) Game-theory branch
    for concept in game_theory:
        for model_short in models_to_short_name.values():
            model_dir = os.path.join(root_dir, 'inferences', concept, model_short)
            if not os.path.isdir(model_dir):
                # nothing to load here
                continue

            for filename in os.listdir(model_dir):
                src = os.path.join(model_dir, filename)
                if not os.path.isfile(src):
                    continue

                with open(src, 'r', encoding='utf-8') as f:
                    raw = f.read()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    print(f"[GameTheory][JSONError] could not parse {src}, skipping")
                    continue

                # normalize correct
                corr = data.get('correct')
                correct_flag = 'yes' if (corr is True or (isinstance(corr, list) and corr and corr[0] is True)) else 'no'

                # build record
                rec = {
                    'Concept': concept,
                    'Correct': correct_flag,
                    'Domain': _get_domain(concept),
                    'File': filename,
                    'Model': model_short,
                    'Task': 'Edit'
                }
                for k, v in data.items():
                    if k in ('concept', 'correct'):
                        continue
                    rec[k] = v

                content = data.get('inferences')
                yield rec, content

    # 2) Literature & psychological-biases branch
    for idx, row in df.iterrows():
        concept     = str(row['Concept']).strip()
        if concept in game_theory:
            continue

        filename    = str(row['File']).strip()
        full_model  = str(row['Model']).strip()
        model_short = models_to_short_name.get(full_model, full_model)
        correct_flag= 'yes' if str(row['Correct']).strip().lower() in ('yes','1','true') else 'no'

        # build record
        rec = row.to_dict()
        rec.update({
            'Concept': concept,
            'Correct': correct_flag,
            'Domain':  _get_domain(concept),
            'File':    filename,
            'Model':   model_short,
            'Task':    'Edit'
        })

        # check model directory
        model_dir = os.path.join(root_dir, 'inferences', concept, model_short)
        if not os.path.isdir(model_dir):
            # print(f"[MissingDir][Row {idx}] Concept={concept!r}, Model={model_short!r}: no dir {model_dir}")
            continue

        # check file
        inf_path = os.path.join(model_dir, filename)
        if not os.path.isfile(inf_path):
            print(f"[MissingFile][Row {idx}] Concept={concept!r}, Model={model_short!r}, File={filename!r}: not found")
            continue

        # load and parse
        try:
            with open(inf_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            content = data.get('inferences')
        except json.JSONDecodeError:
            print(f"[JSONError][Row {idx}] could not parse JSON in {inf_path}")
            continue

        yield rec, content
