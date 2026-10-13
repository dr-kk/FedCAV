#!/usr/bin/env python3
"""E32: paired, seed-matched UCF101 statistics from E30/E31 reported results.
Requires numpy and scipy. No training or GPU required.
"""
import csv
from pathlib import Path
import numpy as np
from scipy import stats

SEEDS = [42, 7, 123]
# Values transcribed from completed E30/E31 console summaries.
DATA = {
    'FedAvg': {'accuracy':[49.957,52.658,54.300], 'macro_f1':[53.357,56.487,56.627], 'forgetting':[48.705,42.369,44.519]},
    'FedProx': {'accuracy':[49.902,52.497,54.644], 'macro_f1':[53.338,56.369,57.014], 'forgetting':[48.706,42.370,43.891]},
    'FixedPrototypeReplay': {'accuracy':[59.975,59.766,60.655], 'macro_f1':[63.013,62.956,63.260], 'forgetting':[35.579,32.050,36.310]},
    'FedCAV': {'accuracy':[61.137,59.136,61.738], 'macro_f1':[64.359,62.533,64.536], 'forgetting':[33.165,31.455,33.835]},
}

OUT = Path(__file__).resolve().parent / 'E32_UCF101_Statistics'
OUT.mkdir(exist_ok=True)

def mean_ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    sd = float(x.std(ddof=1))
    half = float(stats.t.ppf(0.975, n-1) * sd / np.sqrt(n))
    return m, sd, m-half, m+half

summary_rows=[]
for method, metrics in DATA.items():
    for metric, values in metrics.items():
        mean,sd,lo,hi = mean_ci(values)
        summary_rows.append({'method':method,'metric':metric,'n':len(values),'mean':mean,'sample_sd':sd,'ci95_low':lo,'ci95_high':hi})

comparisons=[]
for baseline in ['FedAvg','FedProx','FixedPrototypeReplay']:
    for metric in ['accuracy','macro_f1','forgetting']:
        a=np.array(DATA['FedCAV'][metric],dtype=float)
        b=np.array(DATA[baseline][metric],dtype=float)
        # For forgetting, positive improvement means lower forgetting.
        d = b-a if metric=='forgetting' else a-b
        mean,sd,lo,hi=mean_ci(d)
        t,p=stats.ttest_rel(a,b)
        # Paired standardized effect size (Cohen's dz), undefined if SD=0.
        dz=float(mean/sd) if sd>0 else float('nan')
        comparisons.append({'baseline':baseline,'metric':metric,'seed42_improvement':d[0],
          'seed7_improvement':d[1],'seed123_improvement':d[2],
          'mean_improvement_pp':mean,'sd_improvement_pp':sd,
          'ci95_low_pp':lo,'ci95_high_pp':hi,'paired_t':float(t),'p_two_sided':float(p),
          'cohens_dz':dz,'n':3})

def write_csv(path, rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

write_csv(OUT/'e32_summary.csv',summary_rows)
write_csv(OUT/'e32_paired_comparisons.csv',comparisons)
with (OUT/'e32_report.txt').open('w',encoding='utf-8') as f:
    f.write('E32 UCF101 paired analysis (n=3; seeds 42,7,123)\n')
    f.write('Values manually transcribed from completed E30/E31 logs.\n')
    f.write('Confidence intervals: two-sided Student t, df=2, unadjusted.\n')
    f.write('p-values: two-sided paired t-tests, unadjusted for multiple comparisons.\n')
    f.write('Small n; do not interpret non-significance as equivalence.\n\n')
    for r in summary_rows:
        f.write(f"{r['method']:22s} {r['metric']:12s} {r['mean']:8.3f} +/- {r['sample_sd']:7.3f}  CI95 [{r['ci95_low']:8.3f}, {r['ci95_high']:8.3f}]\n")
    f.write('\nPaired FedCAV improvement (positive = better):\n')
    for r in comparisons:
        f.write(f"vs {r['baseline']:22s} {r['metric']:12s} improvement {r['mean_improvement_pp']:+7.3f} pp, CI95 [{r['ci95_low_pp']:+8.3f}, {r['ci95_high_pp']:+8.3f}], p={r['p_two_sided']:.5f}\n")

print((OUT/'e32_report.txt').read_text(encoding='utf-8'))
