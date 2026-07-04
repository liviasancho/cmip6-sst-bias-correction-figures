#!/usr/bin/env python
# coding: utf-8

# # This code evaluate and diagnose the bias type (constant, linear or nonlinear) for each pair area-model according to the observations

# In[ ]:


import numpy as np
import pandas as pd
import os
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from scipy.stats import linregress
from pathlib import Path


# In[ ]:


# Repository root. This works both when running from the repository root
# (python scripts/script_name.py) and from inside scripts/.
try:
    ROOT_PATH = Path(__file__).resolve().parents[1]
except NameError:
    ROOT_PATH = Path.cwd()

DATA_PATH = ROOT_PATH / "input_data"
OUTPUT_PATH = ROOT_PATH / "out_sst"
TABLE_PATH = ROOT_PATH / "tables"
TABLE_METRICS_PATH = TABLE_PATH / "metrics"

FIGURE_PATH = ROOT_PATH / "figures"
FIGURE_DIAG_PATH = FIGURE_PATH / "bias_diagnostics"
FIGURE_DIAG_SCATTER_MODEL_PATH = FIGURE_DIAG_PATH / "scatter_obs_model"
FIGURE_DIAG_SCATTER_BIAS_PATH = FIGURE_DIAG_PATH / "scatter_obs_bias"
FIGURE_DIAG_QUANTILE_BIAS_PATH = FIGURE_DIAG_PATH / "quantile_bias"
FIGURE_DIAG_SUMMARY_PATH = FIGURE_DIAG_PATH / "summary"

for path in [
    DATA_PATH,
    OUTPUT_PATH,
    TABLE_METRICS_PATH,
    FIGURE_DIAG_PATH,
    FIGURE_DIAG_SCATTER_MODEL_PATH,
    FIGURE_DIAG_SCATTER_BIAS_PATH,
    FIGURE_DIAG_QUANTILE_BIAS_PATH,
    FIGURE_DIAG_SUMMARY_PATH,
]:
    path.mkdir(parents=True, exist_ok=True)


# ### Models and areas to be used

# In[79]:


models = ['INM-CM5-0', 'MIROC6', 'EC-EARTH3-Veg', 'CMCC-ESM2', 'ACCESS-CM2', 'BCC-CSM2-MR']


# Internal names as they appear in the data
areas = [
    'nino12',
    'nino3',
    'nino34',
    'nino4',
    'norte',
    'norteleste',
    'centro',
    'centroleste',
    'cbm',
    'tna',
    'tsa',
    'nat',
    'sat',
    'tasi',
    'wtio',
    'setio',
    'swio',
    'dmi'
]

# Names used in figure titles, labels and tables
area_display_name_map = {
    'nino12': 'Niño 1+2',
    'nino3': 'Niño 3',
    'nino34': 'Niño 3.4',
    'nino4': 'Niño 4',
    'norte': 'SNA_N',
    'norteleste': 'SNA_NE',
    'centro': 'SNA_C',
    'centroleste': 'SNA_CE',
    'cbm': 'BMC',
    'tna': 'TNA',
    'tsa': 'TSA',
    'nat': 'NAT',
    'sat': 'SAT',
    'tasi': 'TASI',
    'wtio': 'WTIO',
    'setio': 'SETIO',
    'swio': 'SWIO',
    'dmi': 'DMI'
}

# Names used in saved file names
area_file_name_map = {
    'nino12': 'nino12',
    'nino3': 'nino3',
    'nino34': 'nino34',
    'nino4': 'nino4',
    'norte': 'sna_n',
    'norteleste': 'sna_ne',
    'centro': 'sna_c',
    'centroleste': 'sna_ce',
    'cbm': 'bmc',
    'tna': 'tna',
    'tsa': 'tsa',
    'nat': 'nat',
    'sat': 'sat',
    'tasi': 'tasi',
    'wtio': 'wtio',
    'setio': 'setio',
    'swio': 'swio',
    'dmi': 'dmi'
}


# ### Loading satellite SST

# In[ ]:


file_path = os.path.join(DATA_PATH, 'merged_sst_series_obs.csv')

sst_obs = pd.read_csv(
    file_path,
    header=None,
    skiprows=1,
    names=['time', 'extra_zero'] + areas
)

sst_obs['time'] = pd.to_datetime(sst_obs['time'])
sst_obs = sst_obs.set_index('time')

for col in ['extra_zero'] + areas:
    sst_obs[col] = pd.to_numeric(sst_obs[col], errors='coerce')

sst_obs = sst_obs.drop(columns='extra_zero')

sst_obs


# ### Load CMIP6 data

# In[ ]:


series_historical = {}

for model in models:
    print(f"\nLoading model: {model}")

    file_path = os.path.join(DATA_PATH, f"{model}_historical_sst_series.csv")

    df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

    # Only the areas of interest are kept
    df = df[areas]
    series_historical[model] = df


# In[82]:


for model in models:
    print(model)
    print("historical:", series_historical[model].index.min(), "->", series_historical[model].index.max())


# # Bias diagnostics

# Alignment function

# In[83]:


def prepare_bias_diagnostic_pair(cmip_series, obs_series,
                                 start='1981-01-01',
                                 end='2014-12-31'):
    cmip = cmip_series.copy()
    obs = obs_series.copy()

    cmip.index = pd.to_datetime(cmip.index)
    obs.index = pd.to_datetime(obs.index)

    cmip = cmip.loc[start:end]
    obs = obs.loc[start:end]

    cmip.index = cmip.index.to_period('M')
    obs.index = obs.index.to_period('M')

    cmip = pd.to_numeric(cmip, errors='coerce')
    obs = pd.to_numeric(obs, errors='coerce')

    df = pd.concat(
        [obs.rename('obs'), cmip.rename('model')],
        axis=1,
        join='inner'
    ).dropna()

    df['bias'] = df['model'] - df['obs']

    return df


# Quantile bias function

# In[84]:


def compute_quantile_bias(df, n_quantiles=10):
    dfq = df.copy()

    dfq['obs_bin'] = pd.qcut(dfq['obs'], q=n_quantiles, duplicates='drop')

    quant_bias = dfq.groupby('obs_bin', observed=False).agg(
        obs_mean=('obs', 'mean'),
        bias_mean=('bias', 'mean'),
        bias_std=('bias', 'std'),
        n=('bias', 'count')
    ).reset_index()

    quant_bias['bias_sem'] = quant_bias['bias_std'] / np.sqrt(quant_bias['n'])

    return quant_bias


# Summary metrics of the bias diagnostics function

# In[85]:


def summarize_bias_structure(df, quant_bias):
    mean_bias = df['bias'].mean()
    std_bias = df['bias'].std()
    corr_obs_bias = df['obs'].corr(df['bias'])

    reg = linregress(df['obs'], df['bias'])

    # bins centrais e extremos
    n_bins = len(quant_bias)

    if n_bins >= 4:
        central = quant_bias.iloc[n_bins//2 - 1:n_bins//2 + 1]['bias_mean'].mean()
        extreme = pd.concat([
            quant_bias.iloc[:2]['bias_mean'],
            quant_bias.iloc[-2:]['bias_mean']
        ]).mean()
    else:
        central = np.nan
        extreme = np.nan

    extreme_minus_central = extreme - central
    bias_range = quant_bias['bias_mean'].max() - quant_bias['bias_mean'].min()

    summary = {
        'mean_bias': mean_bias,
        'std_bias': std_bias,
        'corr_obs_bias': corr_obs_bias,
        'bias_vs_obs_slope': reg.slope,
        'bias_vs_obs_rvalue': reg.rvalue,
        'bias_vs_obs_pvalue': reg.pvalue,
        'central_bias_mean': central,
        'extreme_bias_mean': extreme,
        'extreme_minus_central': extreme_minus_central,
        'quantile_bias_range': bias_range,
        'n_samples': len(df)
    }

    return summary


# Main loop

# In[ ]:


bias_diag_data = {}
bias_diag_quantiles = {}
bias_diag_summary = {}

for model in models:
    print(f'\nDiagnosing bias: {model}')

    bias_diag_data[model] = {}
    bias_diag_quantiles[model] = {}
    bias_diag_summary[model] = {}

    for area in areas:
        df = prepare_bias_diagnostic_pair(
            series_historical[model][area],
            sst_obs[area],
            start='1981-01-01',
            end='2014-12-31'
        )

        quant_bias = compute_quantile_bias(df, n_quantiles=10)
        summary = summarize_bias_structure(df, quant_bias)

        bias_diag_data[model][area] = df
        bias_diag_quantiles[model][area] = quant_bias
        bias_diag_summary[model][area] = summary


# In[87]:


# ============================================================
# Area-specific axis limits for all diagnostic figures
# Same limits within each area across all models
# ============================================================

def add_padding(vmin, vmax, pad_frac=0.05, min_pad=0.1):
    """
    Add proportional padding to axis limits.
    """
    data_range = vmax - vmin
    pad = max(data_range * pad_frac, min_pad)
    return vmin - pad, vmax + pad


# Dictionaries with one set of limits per area.
# Each area keeps the same limits across all models,
# but different areas may have different limits.
obs_model_limits_by_area = {}
obs_limits_by_area = {}
bias_limits_by_area = {}
quant_obs_limits_by_area = {}
quant_bias_limits_by_area = {}

for area in areas:
    # --------------------------------------------------------
    # 1) Observed vs model scatter
    # Same x and y limits for all models within this area
    # --------------------------------------------------------
    area_obs_model_values = []

    for model in models:
        df = bias_diag_data[model][area]
        area_obs_model_values.extend(df['obs'].dropna().values)
        area_obs_model_values.extend(df['model'].dropna().values)

    obs_model_min = np.nanmin(area_obs_model_values)
    obs_model_max = np.nanmax(area_obs_model_values)
    obs_model_limits_by_area[area] = add_padding(obs_model_min, obs_model_max, pad_frac=0.05)

    # --------------------------------------------------------
    # 2) Observed vs bias scatter
    # Same x and y limits for all models within this area
    # --------------------------------------------------------
    area_obs_values = []
    area_bias_values = []

    for model in models:
        df = bias_diag_data[model][area]
        area_obs_values.extend(df['obs'].dropna().values)
        area_bias_values.extend(df['bias'].dropna().values)

    obs_min = np.nanmin(area_obs_values)
    obs_max = np.nanmax(area_obs_values)
    bias_min = np.nanmin(area_bias_values)
    bias_max = np.nanmax(area_bias_values)

    obs_limits_by_area[area] = add_padding(obs_min, obs_max, pad_frac=0.05)
    bias_limits_by_area[area] = add_padding(bias_min, bias_max, pad_frac=0.05)

    # --------------------------------------------------------
    # 3) Quantile bias plot
    # Same x and y limits for all models within this area
    # Includes error bars in y-axis limits
    # --------------------------------------------------------
    area_quant_obs_values = []
    area_quant_bias_values = []

    for model in models:
        quant_bias = bias_diag_quantiles[model][area]

        area_quant_obs_values.extend(quant_bias['obs_mean'].dropna().values)

        y_lower = quant_bias['bias_mean'] - quant_bias['bias_sem']
        y_upper = quant_bias['bias_mean'] + quant_bias['bias_sem']

        area_quant_bias_values.extend(y_lower.dropna().values)
        area_quant_bias_values.extend(y_upper.dropna().values)

    quant_obs_min = np.nanmin(area_quant_obs_values)
    quant_obs_max = np.nanmax(area_quant_obs_values)
    quant_bias_min = np.nanmin(area_quant_bias_values)
    quant_bias_max = np.nanmax(area_quant_bias_values)

    quant_obs_limits_by_area[area] = add_padding(quant_obs_min, quant_obs_max, pad_frac=0.05)
    quant_bias_limits_by_area[area] = add_padding(quant_bias_min, quant_bias_max, pad_frac=0.05)


print('Area-specific axis limits defined:')
for area in areas:
    nome = area_display_name_map[area]
    print(f'\n{nome} ({area})')
    print(f'  Observed/model scatter: xlim = ylim = {obs_model_limits_by_area[area]}')
    print(f'  Observed vs bias scatter: xlim = {obs_limits_by_area[area]}, ylim = {bias_limits_by_area[area]}')
    print(f'  Quantile bias plot: xlim = {quant_obs_limits_by_area[area]}, ylim = {quant_bias_limits_by_area[area]}')


# Final summary table

# In[ ]:


rows = []

for model in models:
    for area in areas:
        row = {'model': model, 'area': area}
        row.update(bias_diag_summary[model][area])
        rows.append(row)

df_bias_diagnostic_summary = pd.DataFrame(rows)
df_bias_diagnostic_summary.to_csv(
    os.path.join(TABLE_METRICS_PATH, 'bias_diagnostic_summary_1981_2014.csv'),
    index=False
)

df_bias_diagnostic_summary


# # Figures

# Scatter obs × model

# In[ ]:


fig_bias_scatter_path = os.path.join(FIGURE_DIAG_PATH, 'scatter_obs_model')
os.makedirs(fig_bias_scatter_path, exist_ok=True)

for model in models:
    for area in areas:
        nome = area_display_name_map[area]
        area_file = area_file_name_map[area]
        df = bias_diag_data[model][area]

        fig, ax = plt.subplots(figsize=(5.5, 5.5))

        ax.scatter(df['obs'], df['model'], alpha=0.5)
        ax.axline((0, 0), slope=1, color='red', linestyle='--')

        # Use the same limits in x and y for this area across all models.
        # This makes the distance from the 1:1 line visually comparable.
        ax.set_xlim(obs_model_limits_by_area[area])
        ax.set_ylim(obs_model_limits_by_area[area])
        ax.set_aspect('equal', adjustable='box')

        ax.set_xlabel('Observed SST (°C)')
        ax.set_ylabel('Historical model SST (°C)')
        ax.set_title(f'{nome} - {model}\nObserved vs model')
        ax.grid(True)

        plt.savefig(
            os.path.join(FIGURE_DIAG_SCATTER_MODEL_PATH, f'{model}_{area_file}_scatter_obs_model.png'),
            format='png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()


# Scatter obs × bias

# In[ ]:


fig_bias_obs_path = os.path.join(FIGURE_DIAG_PATH, 'scatter_obs_bias')
os.makedirs(fig_bias_obs_path, exist_ok=True)

for model in models:
    for area in areas:
        nome = area_display_name_map[area]
        area_file = area_file_name_map[area]
        df = bias_diag_data[model][area]
        summary = bias_diag_summary[model][area]

        fig, ax = plt.subplots(figsize=(5.5, 5.5))

        ax.scatter(df['obs'], df['bias'], alpha=0.5)
        ax.axhline(0, color='red', linestyle='--')

        # Use the same observed-SST and bias limits for this area across all models.
        ax.set_xlim(obs_limits_by_area[area])
        ax.set_ylim(bias_limits_by_area[area])

        ax.set_xlabel('Observed SST (°C)')
        ax.set_ylabel('Bias = model - obs (°C)')
        ax.set_title(f'{nome} - {model}\nObserved vs bias')
        ax.grid(True)

        plt.savefig(
            os.path.join(FIGURE_DIAG_SCATTER_BIAS_PATH, f'{model}_{area_file}_scatter_obs_bias.png'),
            format='png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()


# Quantile bias

# In[ ]:


fig_bias_quant_path = os.path.join(FIGURE_DIAG_PATH, 'quantile_bias')
os.makedirs(fig_bias_quant_path, exist_ok=True)

for model in models:
    for area in areas:
        nome = area_display_name_map[area]
        area_file = area_file_name_map[area]
        quant_bias = bias_diag_quantiles[model][area]

        plt.figure(figsize=(7, 4.5))
        plt.errorbar(
            quant_bias['obs_mean'],
            quant_bias['bias_mean'],
            yerr=quant_bias['bias_sem'],
            marker='o',
            capsize=4
        )
        plt.axhline(0, color='red', linestyle='--')

        # Use the same quantile-SST and quantile-bias limits for this area across all models.
        plt.xlim(quant_obs_limits_by_area[area])
        plt.ylim(quant_bias_limits_by_area[area])

        plt.xlabel('Observed SST by quantile bin (°C)')
        plt.ylabel('Mean bias (°C)')
        plt.title(f'{nome} - {model}\nBias by observed quantiles')
        plt.grid(True)

        plt.savefig(
            os.path.join(FIGURE_DIAG_QUANTILE_BIAS_PATH, f'{model}_{area_file}_quantile_bias.png'),
            format='png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()


# In[92]:


def classify_bias_type(row,
                       slope_small=0.10,
                       corr_small=0.20,
                       quant_range_small=0.50,
                       quant_range_large=1.00,
                       extreme_center_large=0.50):
    """
    Diagnostic classification of bias structure.

    Returns
    -------
    str
        One of:
        - 'constant-like'
        - 'linear-like'
        - 'nonlinear-like'
    """
    slope = abs(row['bias_vs_obs_slope'])
    corr = abs(row['corr_obs_bias'])
    qrange = abs(row['quantile_bias_range'])
    ext_cent = abs(row['extreme_minus_central'])

    # Nearly constant bias
    if (slope < slope_small) and (corr < corr_small) and (qrange < quant_range_small):
        return 'constant-like'

    # Strong distribution-dependent / nonlinear behavior
    if (qrange >= quant_range_large) or (ext_cent >= extreme_center_large):
        return 'nonlinear-like'

    # Otherwise, predominantly linear-like
    return 'linear-like'


# In[93]:


df_bias_diagnostic_summary['bias_type'] = df_bias_diagnostic_summary.apply(
    classify_bias_type,
    axis=1
)


# Heatmap of the predominant bias by model-area

# In[94]:


bias_type_code_map = {
    'constant-like': 0,
    'linear-like': 1,
    'nonlinear-like': 2
}

bias_type_label_map = {
    0: 'Constant-like',
    1: 'Linear-like',
    2: 'Nonlinear-like'
}


# In[95]:


# bias_heatmap_df = df_bias_diagnostic_summary.pivot(
#     index='area',
#     columns='model',
#     values='bias_type'
# )

area_label_map = dict(zip(areas2, nomes_areas))

bias_heatmap_df = df_bias_diagnostic_summary.copy()
bias_heatmap_df['area_label'] = bias_heatmap_df['area'].map(area_label_map)

bias_heatmap_df = bias_heatmap_df.pivot(
    index='area_label',
    columns='model',
    values='bias_type'
)

# Se quiser, renomeie aqui com seus nomes bonitos
# bias_heatmap_df = bias_heatmap_df.rename(index=area_label_map)

bias_heatmap_num = bias_heatmap_df.replace(bias_type_code_map)


# In[ ]:


cmap_bias = ListedColormap(['#8172B2', '#4C72B0', '#DD8452'])
norm_bias = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap_bias.N)


# In[ ]:


fig, ax = plt.subplots(figsize=(max(6, len(bias_heatmap_num.columns) * 0.55),
                                max(3.5, len(bias_heatmap_num.index) * 0.42)))

im = ax.imshow(bias_heatmap_num.values, cmap=cmap_bias, norm=norm_bias, aspect='auto')

ax.set_xticks(np.arange(len(bias_heatmap_num.columns)))
ax.set_yticks(np.arange(len(bias_heatmap_num.index)))

ax.set_xticklabels(bias_heatmap_num.columns, rotation=45, ha='right')
ax.set_yticklabels(bias_heatmap_num.index)

short_label_map = {
    'constant-like': 'Const.',
    'linear-like': 'Linear',
    'nonlinear-like': 'Nonlin.'
}

for i in range(len(bias_heatmap_num.index)):
    for j in range(len(bias_heatmap_num.columns)):
        label = short_label_map.get(bias_heatmap_df.iloc[i, j], bias_heatmap_df.iloc[i, j])
        ax.text(j, i, label, ha='center', va='center', fontsize=8)

# legend_elements = [
#     Patch(facecolor='#8172B2', edgecolor='none', label='Constant-like'),
#     Patch(facecolor='#4C72B0', edgecolor='none', label='Linear-like'),
#     Patch(facecolor='#DD8452', edgecolor='none', label='Nonlinear-like')
# ]

# ax.legend(
#     handles=legend_elements,
#     loc='upper center',
#     bbox_to_anchor=(0.5, -0.12),
#     ncol=3,
#     frameon=False
# )

ax.set_title('Predominant bias structure by model and area')
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIAG_SUMMARY_PATH, 'heatmap_bias_type_by_model_area.png'),
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# In[ ]:


box_data = [
    df_bias_diagnostic_summary.loc[
        df_bias_diagnostic_summary['bias_type'] == btype,
        'quantile_bias_range'
    ].dropna().values
    for btype in ['constant-like', 'linear-like', 'nonlinear-like']
]


# In[ ]:


plt.figure(figsize=(7, 4))
bp = plt.boxplot(
    box_data,
    labels=['Constant-like', 'Linear-like', 'Nonlinear-like'],
    patch_artist=True
)

box_colors = ['#8172B2', '#4C72B0', '#DD8452']
for patch, color in zip(bp['boxes'], box_colors):
    patch.set_facecolor(color)

plt.ylabel('Quantile bias range (°C)')
plt.title('Distribution of quantile-bias range by diagnosed bias type')
plt.grid(True)

plt.savefig(
    os.path.join(FIGURE_DIAG_SUMMARY_PATH, 'boxplot_quantile_bias_range_by_bias_type.png'),
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()

