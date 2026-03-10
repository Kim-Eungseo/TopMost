"""
Update paper tables with actual experiment results from results.json.
Run this script after all experiments complete to fill in the placeholder values.
"""

import sys, os, json, re

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json')
PAPER_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'paper', 'ebmltm_paper.tex')

LV_GRID = [0.01, 0.1, 0.5, 1.0]
LC_GRID = [0.01, 0.1, 0.5]
DELTA_V_GRID = [0.01, 0.02, 0.05, 0.1, 0.2]
DATASETS = ['20NG', 'NYT', 'IMDB']
MODELS = ['ETM', 'ECRTM', 'ETM+Langevin', 'EBMLTM']


def load_results():
    with open(RESULTS_FILE) as f:
        return json.load(f)


def fmt(v, fmt_str='.4f', fallback='---'):
    if v is None or (isinstance(v, float) and v != v):
        return fallback
    try:
        return format(float(v), fmt_str)
    except:
        return fallback


def fmt2(v, fallback='---'):
    return fmt(v, '.2f', fallback)


def build_vicreg_table(results):
    """Build VICReg ablation table rows."""
    lines = []
    best_score = -float('inf')
    best_config = None

    for lv in LV_GRID:
        for lc in LC_GRID:
            k = f'vicreg_{lv}_{lc}_20NG'
            r = results.get(k, {})
            if 'error' in r or 'NPMI' not in r:
                line = f"  {lv} & {lc} & \\multicolumn{{5}}{{c}}{{\\textit{{(error or missing)}}}} \\\\"
            else:
                npmi = r['NPMI']
                td = r['TD']
                nxtd = r.get('NPMI_x_TD', npmi * td)
                acc = r['Acc']
                au = r['AU']
                line = (f"  {lv} & {lc} & ${fmt(npmi)}$ & ${fmt(td)}$ & "
                        f"${fmt(nxtd)}$ & ${fmt(acc)}$ & {au} \\\\")
                score = nxtd + 0.001 * au
                if score > best_score:
                    best_score = score
                    best_config = (lv, lc)
            lines.append(line)

    return '\n'.join(lines), best_config


BASELINE_AGG = {
    '20NG': {'NPMI': '0.0277', 'NPMI_std': '0.0030', 'TD': '0.616', 'TD_std': '0.003',
             'Acc': '0.473', 'Acc_std': '0.013', 'AU': '0.0', 'AU_std': '0.0'},
    'NYT':  {'NPMI': '0.0372', 'NPMI_std': '0.0075', 'TD': '0.646', 'TD_std': '0.011',
             'Acc': '0.673', 'Acc_std': '0.011', 'AU': '0.0', 'AU_std': '0.0'},
}


def build_vicreg_multiseed_table(results):
    """Build VICReg best config multi-seed table rows (all 4 rows: baseline + VICReg per dataset)."""
    lines = []
    for ds_name in ['20NG', 'NYT']:
        b = BASELINE_AGG[ds_name]
        lines.append(
            f"  {ds_name} & EBM-LTM (no VICReg) & "
            f"${b['NPMI']}\\pm{b['NPMI_std']}$ & "
            f"${b['TD']}\\pm{b['TD_std']}$ & "
            f"${b['Acc']}\\pm{b['Acc_std']}$ & $0.0\\pm0.0$ \\\\"
        )
        agg = results.get(f'vicreg_agg_{ds_name}', {})
        if agg:
            nm = agg.get('NPMI_mean', float('nan'))
            ns = agg.get('NPMI_std', float('nan'))
            tm = agg.get('TD_mean', float('nan'))
            ts = agg.get('TD_std', float('nan'))
            am = agg.get('Acc_mean', float('nan'))
            as_ = agg.get('Acc_std', float('nan'))
            au_m = agg.get('AU_mean', float('nan'))
            au_s = agg.get('AU_std', float('nan'))
            lines.append(
                f"  {ds_name} & EBM-LTM + VICReg & "
                f"${fmt(nm)}\\pm{fmt(ns)}$ & "
                f"${fmt(tm)}\\pm{fmt(ts)}$ & "
                f"${fmt(am)}\\pm{fmt(as_)}$ & ${fmt(au_m, '.1f')}\\pm{fmt(au_s, '.1f')}$ \\\\"
            )
        else:
            lines.append(
                f"  {ds_name} & EBM-LTM + VICReg & \\multicolumn{{4}}{{c}}{{\\textit{{(pending)}}}} \\\\"
            )
    return '\n'.join(lines)


def build_stepsize_table(results):
    """Build step size ablation table rows."""
    lines = []
    for delta in DELTA_V_GRID:
        k = f'stepsize_{delta}_20NG'
        r = results.get(k, {})
        if 'error' in r or 'NPMI' not in r:
            line = f"  {delta} & \\multicolumn{{5}}{{c}}{{\\textit{{(pending)}}}} \\\\"
        else:
            npmi = fmt(r['NPMI'])
            td = fmt(r['TD'])
            au = r['AU']
            acc = fmt(r['Acc'])
            ed = fmt(r.get('explore_dist', float('nan')), '.6f')
            line = f"  {delta} & ${npmi}$ & ${td}$ & {au} & ${acc}$ & ${ed}$ \\\\"
        lines.append(line)
    return '\n'.join(lines)


def build_ppl_table(results):
    """Build held-out perplexity table rows."""
    lines = []
    for ds_name in DATASETS:
        lines.append(f"\\multicolumn{{4}}{{l}}{{\\textit{{{ds_name}}}}} \\\\")
        for model_name in MODELS:
            k = f'ppl_{model_name}_{ds_name}'
            r = results.get(k, {})
            display_name = model_name.replace('+', '+')
            if 'error' in r or 'ppl_amortized' not in r:
                line = f"  & {display_name} & \\textit{{pending}} & \\textit{{pending}} \\\\"
            else:
                pa = fmt2(r['ppl_amortized'])
                pr = r.get('ppl_refined')
                pr_str = fmt2(pr) if pr is not None else '---'
                line = f"  & {display_name} & ${pa}$ & ${pr_str}$ \\\\"
            lines.append(line)
        for m in ['BERTopic', 'FASTopic']:
            lines.append(f"  & {m} & --- & --- \\\\")
        lines.append('\\hline')
    return '\n'.join(lines)


def update_vicreg_table_in_paper(paper_text, results):
    """Replace VICReg ablation table body (between markers) with actual results."""
    table_rows, best_config = build_vicreg_table(results)

    # Use explicit marker comments for reliable replacement
    pattern = r'(%VICREG_DATA_START\n)(.*?)(%VICREG_DATA_END)'
    match = re.search(pattern, paper_text, re.DOTALL)
    if not match:
        print("WARNING: Could not find VICREG_DATA_START/END markers")
        return paper_text, best_config

    new_body = table_rows + '\n'
    paper_text = (paper_text[:match.start(2)] + new_body + paper_text[match.end(2):])
    return paper_text, best_config


def update_vicreg_multiseed_table_in_paper(paper_text, results):
    """Replace VICReg multiseed table body (between markers) with actual results."""
    table_rows = build_vicreg_multiseed_table(results)
    pattern = r'(%VICREG_MULTI_DATA_START\n)(.*?)(%VICREG_MULTI_DATA_END)'
    match = re.search(pattern, paper_text, re.DOTALL)
    if not match:
        print("WARNING: Could not find VICREG_MULTI_DATA_START/END markers")
        return paper_text
    paper_text = paper_text[:match.start(2)] + table_rows + '\n' + paper_text[match.end(2):]
    return paper_text


def update_stepsize_table_in_paper(paper_text, results):
    """Replace step size table body (between markers) with actual results."""
    table_rows = build_stepsize_table(results)
    pattern = r'(%STEPSIZE_DATA_START\n)(.*?)(%STEPSIZE_DATA_END)'
    match = re.search(pattern, paper_text, re.DOTALL)
    if not match:
        print("WARNING: Could not find STEPSIZE_DATA_START/END markers")
        return paper_text
    paper_text = paper_text[:match.start(2)] + table_rows + '\n' + paper_text[match.end(2):]
    return paper_text


def update_ppl_table_in_paper(paper_text, results):
    """Replace PPL table body (between markers) with actual results."""
    table_rows = build_ppl_table(results)
    pattern = r'(%PPL_DATA_START\n)(.*?)(%PPL_DATA_END)'
    match = re.search(pattern, paper_text, re.DOTALL)
    if not match:
        print("WARNING: Could not find PPL_DATA_START/END markers")
        return paper_text
    paper_text = paper_text[:match.start(2)] + table_rows + '\n' + paper_text[match.end(2):]
    return paper_text


def print_summary(results):
    print("\n=== VICREG GRID (20NG) ===")
    for lv in LV_GRID:
        for lc in LC_GRID:
            k = f'vicreg_{lv}_{lc}_20NG'
            r = results.get(k, {})
            if 'NPMI' in r:
                print(f"  lv={lv} lc={lc}: NPMI={r['NPMI']:.4f} TD={r['TD']:.4f} AU={r['AU']} Acc={r['Acc']:.4f}")
            else:
                print(f"  lv={lv} lc={lc}: PENDING/ERROR")

    print("\n=== STEPSIZE ABLATION ===")
    for delta in DELTA_V_GRID:
        k = f'stepsize_{delta}_20NG'
        r = results.get(k, {})
        if 'NPMI' in r:
            print(f"  delta={delta}: NPMI={r['NPMI']:.4f} TD={r['TD']:.4f} AU={r['AU']} explore={r.get('explore_dist','?'):.4f}")
        else:
            print(f"  delta={delta}: PENDING/ERROR")

    print("\n=== PPL ===")
    for ds in DATASETS:
        for model in MODELS:
            k = f'ppl_{model}_{ds}'
            r = results.get(k, {})
            if 'ppl_amortized' in r:
                print(f"  {model} {ds}: ppl_amort={r['ppl_amortized']} ppl_refined={r.get('ppl_refined','---')}")
            else:
                print(f"  {model} {ds}: PENDING/ERROR")


def main():
    results = load_results()
    print_summary(results)

    with open(PAPER_FILE, 'r') as f:
        paper_text = f.read()

    # Check if any results are available to update
    has_vicreg = any(f'vicreg_{lv}_{lc}_20NG' in results and 'NPMI' in results[f'vicreg_{lv}_{lc}_20NG']
                     for lv in LV_GRID for lc in LC_GRID)
    has_stepsize = any(f'stepsize_{d}_20NG' in results and 'NPMI' in results[f'stepsize_{d}_20NG']
                       for d in DELTA_V_GRID)
    has_ppl = any(f'ppl_{m}_{ds}' in results and 'ppl_amortized' in results[f'ppl_{m}_{ds}']
                  for m in MODELS for ds in DATASETS)

    has_vicreg_multi = any(f'vicreg_agg_{ds}' in results for ds in ['20NG', 'NYT'])

    updated = False
    if has_vicreg:
        paper_text, best_config = update_vicreg_table_in_paper(paper_text, results)
        print(f"\nVICReg table updated (best config: {best_config})")
        updated = True

    # Always update multiseed table (includes hardcoded baselines + pending/actual VICReg rows)
    paper_text = update_vicreg_multiseed_table_in_paper(paper_text, results)
    if has_vicreg_multi:
        print("VICReg multiseed table updated")
    updated = True

    if has_stepsize:
        paper_text = update_stepsize_table_in_paper(paper_text, results)
        print("Step size table updated")
        updated = True

    if has_ppl:
        paper_text = update_ppl_table_in_paper(paper_text, results)
        print("PPL table updated")
        updated = True

    if updated:
        with open(PAPER_FILE, 'w') as f:
            f.write(paper_text)
        print(f"Paper updated: {PAPER_FILE}")
    else:
        print("No results available yet; paper not modified.")


if __name__ == '__main__':
    main()
