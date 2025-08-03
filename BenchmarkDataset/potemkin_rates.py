import math
import pandas as pd
from iterators import define_iterator, classify_iterator, generate_iterator, edit_iterator


def collect_records():
    """
    Collects metadata dicts from all iterators into a single DataFrame.
    """
    records = []
    for iterator in (define_iterator, classify_iterator, generate_iterator, edit_iterator):
        for meta, _ in iterator():
            records.append(meta)
    return pd.DataFrame(records)


def print_potemkin_rate_by_task():
    """
    Potemkin rate = 1 − accuracy(task | define_correct), expressed as a percentage.
    For Classify (chance accuracy = 0.5), we scale by 2 so that chance-level → 100%.
    Prints ±2×SE for Classify and ±1×SE for others, with raw counts in their own aligned column.
    Also, for Classify, prints the total number of entries per domain.
    """
    # 1) Keystone successes
    define_success = {
        (m['Concept'], m['Model'])
        for m, _ in define_iterator()
        if m['Correct'].strip().lower() == 'yes'
    }

    task_map = {
        'Classify': classify_iterator,
        'Generate': generate_iterator,
        'Edit':     edit_iterator,
    }

    # adjust these to taste
    name_w   = 15
    value_w  = 24
    count_w  = 10

    # header
    print("\nPotemkin rate by task (1 − accuracy | define_correct) [×2 for classify]:\n")
    print(f"{'Task/Model':<{name_w}}{'Potemkin Rate ± SE':<{value_w}}{'Counts':>{count_w}}")
    print("-" * (name_w + value_w + count_w))

    for task_name, task_iter in task_map.items():
        total = correct = 0
        model_stats = {}
        domain_stats = {}

        # gather stats
        for meta, _ in task_iter():
            key = (meta['Concept'], meta['Model'])
            if key not in define_success:
                continue

            total += 1
            domain = meta['Domain']
            domain_stats[domain] = domain_stats.get(domain, 0) + 1

            is_corr = (meta['Correct'].strip().lower() == 'yes')
            if is_corr:
                correct += 1

            m = meta['Model']
            c_cnt, t_cnt = model_stats.get(m, (0, 0))
            model_stats[m] = (c_cnt + (1 if is_corr else 0), t_cnt + 1)

        # overall line
        if total == 0:
            name_str, value_str, counts_str = task_name, "N/A", ""
        else:
            p      = correct / total
            se     = math.sqrt(p * (1 - p) / total)
            mult   = 2 if task_name == 'Classify' else 1
            margin = mult * se * 100

            if task_name == 'Classify':
                rate = (1 - p) * 2 * 100
                note = " (scaled)"
            else:
                rate = (1 - p) * 100
                note = ""

            name_str   = task_name
            value_str  = f"{rate:6.2f}%±{margin:4.1f}%{note}"
            counts_str = f"{correct}/{total}"

        print(f"{name_str:<{name_w}}{value_str:<{value_w}}{counts_str:>{count_w}}")


        # per-model lines
        for model in sorted(model_stats):
            m_corr, m_tot = model_stats[model]
            if m_tot == 0:
                val_str, cnts_str = "N/A", ""
            else:
                p_m      = m_corr / m_tot
                se_m     = math.sqrt(p_m * (1 - p_m) / m_tot)
                mult_m   = 2 if task_name == 'Classify' else 1
                margin_m = mult_m * se_m * 100
                rate_m   = (1 - p_m) * mult_m * 100
                val_str  = f"{rate_m:6.2f}%±{margin_m:4.1f}%"
                cnts_str = f"{m_corr}/{m_tot}"

            name_model = "  " + model
            print(f"{name_model:<{name_w}}{val_str:<{value_w}}{cnts_str:>{count_w}}")

        print()  # blank line between tasks

if __name__ == '__main__':
    print_potemkin_rate_by_task()