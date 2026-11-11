"""Figures for the second-revision analysis, drawn from results/*.csv (run reanalyze.py first).

Outputs in figures/:
  fig1_fidelity.png          out-of-fold OLS benchmark vs. LLM representations (MAE, r, SD ratio)
  fig2_format_components.png corrected C-B contrast variance components
  fig3_order_warming.png     reversed minus five-draw original-order mean
  fig4_adjusted_mirror.png   model contrast magnitude / regression-adjusted human contrast
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
F = ROOT / 'figures'
F.mkdir(exist_ok=True)
names = {'haiku': 'Claude Haiku 4.5', 'ministral': 'Ministral 3B', 'qwen': 'Qwen3 32B', 'gptoss': 'GPT-OSS-120B'}


def read(n):
    return pd.read_csv(R / (n + '.csv'))


fid, mir, fmt, order = read('fidelity'), read('adjusted_mirror'), read('format_decomposition'), read('order_effects')
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150})

# Figure 1: LLM estimates with person-cluster bootstrap intervals; dashed line = out-of-fold OLS.
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), layout='constrained')
keys = [m + '/' + c for m in names for c in 'BC']
labels = [names[m] + ' ' + c for m in names for c in 'BC']
z = fid.set_index('predictor').loc[keys]
for ax, k, title in zip(axs, ['mae', 'r', 'sd_pct'],
                        ['Individual prediction error (MAE)', 'Mean per-item correlation', 'Mean per-item SD (% of human)']):
    val, lo, hi = z[k].to_numpy(), z[k + '_lo'].to_numpy(), z[k + '_hi'].to_numpy()
    ax.errorbar(val, np.arange(8), xerr=[val - lo, hi - val], fmt='o', color='#247ba0', capsize=2)
    ax.axvline(fid.set_index('predictor').loc['OLS (OOF)', k], color='#b34a29', linestyle='--', label='OOF OLS')
    ax.set_yticks(np.arange(8), labels if ax is axs[0] else [''] * 8)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.grid(axis='x', alpha=.2)
axs[0].legend()
fig.savefig(F / 'fig1_fidelity.png', bbox_inches='tight')
plt.close(fig)

# Figure 2: stacked corrected variance components of the C-B contrast (contrast scale, points^2).
fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
bottom = np.zeros(4)
for comp, color in zip(['respondent', 'item', 'respondent_item'], ['#214e65', '#4b8da0', '#93c5c6']):
    v = fmt[fmt.component == comp].set_index('model').loc[list(names), 'value'].to_numpy()
    ax.bar(np.arange(4), v, bottom=bottom, label=comp.replace('_', ' × '), color=color)
    bottom += v
ax.set_xticks(np.arange(4), names.values())
ax.set_ylabel('Corrected contrast variance (points²)')
ax.legend()
fig.savefig(F / 'fig2_format_components.png', bbox_inches='tight')
plt.close(fig)

# Figure 3: signed order shift with 95% person-cluster bootstrap intervals.
fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
for j, c in enumerate('BC'):
    z = order[order.condition == c].set_index('model').loc[list(names)]
    v = z.signed_mean.to_numpy()
    ax.errorbar(np.arange(4) + (j - .5) * .16, v, yerr=[v - z.lo.to_numpy(), z.hi.to_numpy() - v], fmt='o', capsize=4, label=c)
ax.axhline(0, color='gray', lw=1)
ax.set_xticks(np.arange(4), names.values())
ax.set_ylabel('Reversed − original mean (points)')
ax.legend(title='Representation')
fig.savefig(F / 'fig3_order_warming.png', bbox_inches='tight')
plt.close(fig)

# Figure 4: ratio of model to adjusted human party-and-ideology contrast magnitude (1 = equal).
fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
z = mir.set_index('model').loc[list(names)]
v = z.ratio_all.to_numpy()
ax.errorbar(np.arange(4), v, yerr=[v - z.ratio_all_lo.to_numpy(), z.ratio_all_hi.to_numpy() - v], fmt='o', capsize=4, color='#247ba0')
ax.axhline(1, color='gray', linestyle='--')
ax.set_xticks(np.arange(4), names.values())
ax.set_ylabel('Model / adjusted human contrast magnitude')
fig.savefig(F / 'fig4_adjusted_mirror.png', bbox_inches='tight')
plt.close(fig)
