#!/usr/bin/env python
# coding: utf-8

# # This code remove CMIP6 SST bias with 3 different methods: linear regression, quantile mapping and XGBoost

# In[ ]:


import xarray as xr
from pathlib import Path
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.interpolate import interp1d
from xgboost import XGBRegressor


# ### Caminhos importantes

# In[ ]:


# Repository root. This works both when running from the repository root
# (python scripts/script_name.py) and from inside scripts/.
try:
    ROOT_PATH = Path(__file__).resolve().parents[1]
except NameError:
    ROOT_PATH = Path.cwd()

DATA_PATH = ROOT_PATH / "input_data"
OUTPUT_PATH = ROOT_PATH / "out_sst"
SPLIT_PATH = OUTPUT_PATH / "split"
TABLE_PATH = ROOT_PATH / "tables"
TABLE_METRICS_PATH = TABLE_PATH / "metrics"
TABLE_SELECTED_METHODS_PATH = TABLE_PATH / "selected_methods"
TABLE_IMPROVEMENT_PATH = TABLE_PATH / "improvement"

FIGURE_PATH = ROOT_PATH / "figures"

FIGURE_TEST_TS_PATH = FIGURE_PATH / "bias_removal" / "historical_test_timeseries"
FIGURE_TEST_CLIM_PATH = FIGURE_PATH / "bias_removal" / "historical_test_climatology"
FIGURE_TEST_CDF_PATH = FIGURE_PATH / "bias_removal" / "historical_test_distribution_cdf"

FIGURE_SSP_TS_PATH = FIGURE_PATH / "bias_removal" / "ssps_timeseries"
FIGURE_SSP_CLIM_PATH = FIGURE_PATH / "bias_removal" / "ssps_climatology"
FIGURE_SSP_CDF_PATH = FIGURE_PATH / "bias_removal" / "ssps_distribution_cdf"

FIGURE_SUMMARY_PATH = FIGURE_PATH / "bias_removal" / "summary"

for path in []:
    os.makedirs(path, exist_ok=True)

for path in [
    DATA_PATH,
    OUTPUT_PATH,
    SPLIT_PATH,
    TABLE_METRICS_PATH,
    TABLE_SELECTED_METHODS_PATH,
    TABLE_IMPROVEMENT_PATH,
    FIGURE_PATH,
    FIGURE_SUMMARY_PATH,
    FIGURE_TEST_TS_PATH, 
    FIGURE_TEST_CLIM_PATH, 
    FIGURE_TEST_CDF_PATH,
    FIGURE_SSP_TS_PATH, 
    FIGURE_SSP_CLIM_PATH, 
    FIGURE_SSP_CDF_PATH, 
    FIGURE_SUMMARY_PATH
]:
    path.mkdir(parents=True, exist_ok=True)


# ### Models and areas to be used

# In[51]:


models = ['INM-CM5-0', 'MIROC6', 'EC-EARTH3-Veg', 'CMCC-ESM2', 'ACCESS-CM2', 'BCC-CSM2-MR']

experiments = ['historical', 'ssp245', 'ssp585']

# Internal names exactly as they appear in the CSV files.
# Do not replace these names with plotting or filename labels.
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

# Labels used in titles, axes, tables and heatmaps.
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

# Labels used only in saved figure filenames.
# These names are GitHub-friendly and avoid Portuguese terms.
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

# Backward-compatible list used by older cells.
nomes_areas = [area_display_name_map[area] for area in areas]

# Backward-compatible aliases used by summary cells.
area_name_map = area_display_name_map.copy()
area_label_map = area_display_name_map.copy()


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


# ### Loading CMIP6 data

# In[ ]:


series_historical = {}
series_ssp = {}

for model in models:
    print(f"\Loading model: {model}")
    series_ssp[model] = {}

    for experiment in experiments:
        file_path = os.path.join(DATA_PATH, f"{model}_{experiment}_sst_series.csv")
        print(f"  Lendo {experiment}")

        df = pd.read_csv(file_path, parse_dates=['time'], index_col='time')

        # garante que só fiquem as áreas de interesse
        df = df[areas]

        if experiment == 'historical':
            series_historical[model] = df
        else:
            series_ssp[model][experiment] = df


# In[55]:


for model in models:
    print(model)
    print("historical:", series_historical[model].index.min(), "->", series_historical[model].index.max())

    for experiment in series_ssp[model]:
        print(experiment, ":", series_ssp[model][experiment].index.min(), "->", series_ssp[model][experiment].index.max())


# # Step 1: train/test with observations and historical experiment

# ### Set-up

# In[56]:


calib_start = '1981-01-01'
calib_end   = '2014-12-31'

train_start = '1981-01-01'
train_end   = '2004-12-31'

test_start  = '2005-01-01'
test_end    = '2014-12-31'


# ### Defining functions

# Monthly alignment + cutting

# In[57]:


def prepare_aligned_pair(cmip_series, obs_series,
                         calib_start='1981-01-01',
                         calib_end='2014-12-31'):
    cmip = cmip_series.copy()
    obs = obs_series.copy()

    cmip.index = pd.to_datetime(cmip.index)
    obs.index = pd.to_datetime(obs.index)

    cmip = cmip.loc[calib_start:calib_end]
    obs = obs.loc[calib_start:calib_end]

    cmip.index = cmip.index.to_period('M')
    obs.index = obs.index.to_period('M')

    cmip = pd.to_numeric(cmip, errors='coerce')
    obs = pd.to_numeric(obs, errors='coerce')

    df = pd.concat(
        [cmip.rename('cmip'), obs.rename('obs')],
        axis=1,
        join='inner'
    ).dropna()

    return df


# Temporal split

# In[58]:


def temporal_train_test_split(df,
                              train_start='1981-01',
                              train_end='2004-12',
                              test_start='2005-01',
                              test_end='2014-12'):
    df_train = df.loc[train_start:train_end].copy()
    df_test = df.loc[test_start:test_end].copy()

    if len(df_train) == 0:
        raise ValueError("Training set is empty.")
    if len(df_test) == 0:
        raise ValueError("Test set is empty.")

    return df_train, df_test


# Metrics

# In[59]:


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'r2': np.nan,
            'bias': np.nan,
            'corr': np.nan,
            'n': 0
        }

    return {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'bias': np.mean(y_pred - y_true),
        'corr': np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else np.nan,
        'n': len(y_true)
    }


# ## Linear regression with split

# fit/apply

# In[60]:


def fit_linear_bc(df_train):
    X = df_train['cmip'].to_numpy().reshape(-1, 1)
    y = df_train['obs'].to_numpy()

    model = LinearRegression().fit(X, y)
    return model

def apply_linear_bc(series, model):
    s = series.copy()
    s = pd.to_numeric(s, errors='coerce')

    pred = pd.Series(index=s.index, dtype=float, name=s.name)
    mask = s.notna()

    pred.loc[mask] = model.predict(s.loc[mask].to_numpy().reshape(-1, 1))
    return pred


# Loop

# In[61]:


linear_models = {}
linear_test_pred = {}
metrics_linear = {}

for model in models:
    linear_models[model] = {}
    linear_test_pred[model] = {}
    metrics_linear[model] = {}

    print(f"\n[Linear] {model}")

    for area in areas:
        df = prepare_aligned_pair(
            series_historical[model][area],
            sst_obs[area],
            calib_start=calib_start,
            calib_end=calib_end
        )

        df_train, df_test = temporal_train_test_split(
            df,
            train_start='1981-01',
            train_end='2004-12',
            test_start='2005-01',
            test_end='2014-12'
        )

        lin_model = fit_linear_bc(df_train)
        linear_models[model][area] = lin_model

        y_pred = lin_model.predict(df_test['cmip'].to_numpy().reshape(-1, 1))
        y_pred = pd.Series(y_pred, index=df_test.index, name=area)

        linear_test_pred[model][area] = y_pred
        metrics_linear[model][area] = compute_metrics(df_test['obs'], y_pred)

        print(f"{area} | train={len(df_train)} | test={len(df_test)}")


# ## Quantile Mapping with split

# fit/apply

# In[62]:


def fit_quantile_mapping(train_model, train_obs, n_quantiles=1001):
    train_model = np.asarray(train_model, dtype=float)
    train_obs = np.asarray(train_obs, dtype=float)

    train_model = train_model[~np.isnan(train_model)]
    train_obs = train_obs[~np.isnan(train_obs)]

    percentiles = np.linspace(0, 100, n_quantiles)

    q_mod = np.percentile(train_model, percentiles)
    q_obs = np.percentile(train_obs, percentiles)

    q_mod_unique, idx = np.unique(q_mod, return_index=True)
    q_obs_unique = q_obs[idx]

    if len(q_mod_unique) < 2:
        raise ValueError("Not enough unique quantiles to fit QM.")

    qm_func = interp1d(
        q_mod_unique,
        q_obs_unique,
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate'
    )
    return qm_func

def apply_quantile_mapping(series, qm_func):
    s = pd.to_numeric(series.copy(), errors='coerce')

    pred = pd.Series(index=s.index, dtype=float, name=s.name)
    mask = s.notna()

    pred.loc[mask] = qm_func(s.loc[mask].to_numpy())
    return pred


# Loop

# In[63]:


qm_funcs = {}
qm_test_pred = {}
metrics_qm = {}

for model in models:
    qm_funcs[model] = {}
    qm_test_pred[model] = {}
    metrics_qm[model] = {}

    print(f"\n[QM] {model}")

    for area in areas:
        df = prepare_aligned_pair(
            series_historical[model][area],
            sst_obs[area],
            calib_start=calib_start,
            calib_end=calib_end
        )

        df_train, df_test = temporal_train_test_split(
            df,
            train_start='1981-01',
            train_end='2004-12',
            test_start='2005-01',
            test_end='2014-12'
        )

        qm_func = fit_quantile_mapping(df_train['cmip'], df_train['obs'])
        qm_funcs[model][area] = qm_func

        y_pred = apply_quantile_mapping(df_test['cmip'], qm_func)

        qm_test_pred[model][area] = y_pred
        metrics_qm[model][area] = compute_metrics(df_test['obs'], y_pred)

        print(f"{area} | train={len(df_train)} | test={len(df_test)}")


# ## XGBoost with split

# fit/apply

# In[64]:


def build_xgb_features_from_series(series):
    s = pd.to_numeric(series.copy(), errors='coerce')

    month = s.index.month
    X = pd.DataFrame({
        'cmip': s,
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12)
    }, index=s.index)

    return X


# In[65]:


def fit_xgb_bc(df_train, random_state=42):
    month = df_train.index.month

    X_train = pd.DataFrame({
        'cmip': df_train['cmip'].to_numpy(),
        'month_sin': np.sin(2 * np.pi * month / 12),
        'month_cos': np.cos(2 * np.pi * month / 12)
    }, index=df_train.index)

    y_train = df_train['obs']

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=random_state
    )

    model.fit(X_train, y_train)
    return model


# In[66]:


def apply_xgb_bc(series, model):
    X = build_xgb_features_from_series(series)

    pred = pd.Series(index=series.index, dtype=float, name=series.name)
    mask = X['cmip'].notna()

    pred.loc[mask] = model.predict(X.loc[mask])
    return pred


# Loop

# In[67]:


xgb_models = {}
xgb_test_pred = {}
metrics_xgb = {}

for model in models:
    xgb_models[model] = {}
    xgb_test_pred[model] = {}
    metrics_xgb[model] = {}

    print(f"\n[XGB] {model}")

    for area in areas:
        df = prepare_aligned_pair(
            series_historical[model][area],
            sst_obs[area],
            calib_start=calib_start,
            calib_end=calib_end
        )

        df_train, df_test = temporal_train_test_split(
            df,
            train_start='1981-01',
            train_end='2004-12',
            test_start='2005-01',
            test_end='2014-12'
        )

        xgb_model = fit_xgb_bc(df_train, random_state=42)
        xgb_models[model][area] = xgb_model

        y_pred = apply_xgb_bc(df_test['cmip'], xgb_model)

        xgb_test_pred[model][area] = y_pred
        metrics_xgb[model][area] = compute_metrics(df_test['obs'], y_pred)

        print(f"{area} | train={len(df_train)} | test={len(df_test)}")


# ### Gathering the metrics

# In[ ]:


rows = []

for model in models:
    for area in areas:
        for method_name, metrics_dict in [
            ('linear', metrics_linear),
            ('qm', metrics_qm),
            ('xgb', metrics_xgb)
        ]:
            row = {
                'model': model,
                'area': area,
                'method': method_name
            }
            row.update(metrics_dict[model][area])
            rows.append(row)

df_metrics_all = pd.DataFrame(rows)
df_metrics_all.to_csv(os.path.join(TABLE_METRICS_PATH, 'bias_correction_metrics_1981_2014_split.csv'), index=False)
df_metrics_all


# ### Historical baseline without correction
# 
# Calculation of the error of the raw historical model in the same test period used to evaluate the bias correction methods.

# In[ ]:


baseline_rows = []

for model in models:
    for area in areas:
        df = prepare_aligned_pair(
            series_historical[model][area],
            sst_obs[area],
            calib_start=calib_start,
            calib_end=calib_end
        )

        _, df_test = temporal_train_test_split(
            df,
            train_start='1981-01',
            train_end='2004-12',
            test_start='2005-01',
            test_end='2014-12'
        )

        baseline_metrics = compute_metrics(
            y_true=df_test['obs'],
            y_pred=df_test['cmip']
        )

        row = {
            'model': model,
            'area': area,
            'method': 'raw_historical'
        }
        row.update(baseline_metrics)
        baseline_rows.append(row)

df_metrics_baseline = pd.DataFrame(baseline_rows)

df_metrics_baseline.to_csv(
    os.path.join(TABLE_METRICS_PATH, 'historical_raw_baseline_metrics_2005_2014.csv'),
    index=False
)

baseline_rmse_median = df_metrics_baseline['rmse'].median()
baseline_mae_median = df_metrics_baseline['mae'].median()

print(f"Raw historical baseline median RMSE: {baseline_rmse_median:.3f}")
print(f"Raw historical baseline median MAE:  {baseline_mae_median:.3f}")

df_metrics_baseline


# # Step 2: Applying the "winning" method to correct the bias.

# Configure priority and threshold.

# In[72]:


method_priority = {
    'linear': 0,
    'qm': 1,
    'xgb': 2
}

rmse_tie_threshold = 0.02
mae_tie_threshold = 0.01
r2_tie_threshold = 0.01


# Choosing fuction

# In[73]:


def select_best_method_group(df_group,
                             rmse_tie_threshold=0.02,
                             mae_tie_threshold=0.01,
                             r2_tie_threshold=0.01,
                             method_priority=None):
    """
    Select the best bias-correction method for one model-area group
    using a robust hierarchical rule.

    Priority:
    1. Lower RMSE
    2. If practical tie in RMSE, lower MAE
    3. If practical tie in MAE, higher R²
    4. If still practically tied, prefer simpler method
    """
    if method_priority is None:
        method_priority = {'linear': 0, 'qm': 1, 'xgb': 2}

    df = df_group.copy()

    # drop rows without essential metrics
    df = df.dropna(subset=['rmse', 'mae', 'r2'])

    if len(df) == 0:
        raise ValueError("No valid methods available in group.")

    # Step 1: best RMSE
    best_rmse = df['rmse'].min()
    df_rmse = df[df['rmse'] <= best_rmse + rmse_tie_threshold].copy()

    # If only one survives, choose it
    if len(df_rmse) == 1:
        chosen = df_rmse.iloc[0].copy()
        chosen['selection_reason'] = 'lowest_rmse'
        return chosen

    # Step 2: best MAE within RMSE tie
    best_mae = df_rmse['mae'].min()
    df_mae = df_rmse[df_rmse['mae'] <= best_mae + mae_tie_threshold].copy()

    if len(df_mae) == 1:
        chosen = df_mae.iloc[0].copy()
        chosen['selection_reason'] = 'rmse_tie_then_lowest_mae'
        return chosen

    # Step 3: best R² within RMSE+MAE tie
    best_r2 = df_mae['r2'].max()
    df_r2 = df_mae[df_mae['r2'] >= best_r2 - r2_tie_threshold].copy()

    if len(df_r2) == 1:
        chosen = df_r2.iloc[0].copy()
        chosen['selection_reason'] = 'rmse_mae_tie_then_highest_r2'
        return chosen

    # Step 4: parsimony
    df_r2['complexity_rank'] = df_r2['method'].map(method_priority)
    df_r2 = df_r2.sort_values(by='complexity_rank', ascending=True)

    chosen = df_r2.iloc[0].copy()
    chosen['selection_reason'] = 'practical_tie_prefer_simpler_method'
    return chosen


# Applying the choice in the table

# In[74]:


selected_rows = []

for (model, area), df_group in df_metrics_all.groupby(['model', 'area']):
    chosen = select_best_method_group(
        df_group,
        rmse_tie_threshold=rmse_tie_threshold,
        mae_tie_threshold=mae_tie_threshold,
        r2_tie_threshold=r2_tie_threshold,
        method_priority=method_priority
    )
    selected_rows.append(chosen)

best_methods_robust = pd.DataFrame(selected_rows).reset_index(drop=True)
best_methods_robust


# In[ ]:


best_methods_robust.to_csv(
    os.path.join(TABLE_SELECTED_METHODS_PATH, 'best_bias_correction_method_robust_selection.csv'),
    index=False
)


# In[76]:


best_method_dict = {
    (row['model'], row['area']): row['method']
    for _, row in best_methods_robust.iterrows()
}


# ## Final Linear Regression

# In[77]:


def fit_linear_full(cmip_series, obs_series,
                    calib_start='1981-01-01',
                    calib_end='2014-12-31'):
    df = prepare_aligned_pair(
        cmip_series,
        obs_series,
        calib_start=calib_start,
        calib_end=calib_end
    )

    model = fit_linear_bc(df)
    return model


# ## Final Quantile Mapping

# In[78]:


def fit_qm_full(cmip_series, obs_series,
                calib_start='1981-01-01',
                calib_end='2014-12-31'):
    df = prepare_aligned_pair(
        cmip_series,
        obs_series,
        calib_start=calib_start,
        calib_end=calib_end
    )

    qm_func = fit_quantile_mapping(df['cmip'], df['obs'])
    return qm_func


# ## Final XGBoost

# In[79]:


def fit_xgb_full(cmip_series, obs_series,
                 calib_start='1981-01-01',
                 calib_end='2014-12-31',
                 random_state=42):
    df = prepare_aligned_pair(
        cmip_series,
        obs_series,
        calib_start=calib_start,
        calib_end=calib_end
    )

    model = fit_xgb_bc(df, random_state=random_state)
    return model


# ## Apply the best method

# In[ ]:


hist_best = {}
ssp_best = {}
best_fitted_objects = {}
best_method_used = {}

for model in models:
    print(f"\nAplicando melhor método final para: {model}")

    hist_best[model] = pd.DataFrame(index=series_historical[model].index)
    ssp_best[model] = {}
    best_fitted_objects[model] = {}
    best_method_used[model] = {}

    for experiment in series_ssp[model]:
        ssp_best[model][experiment] = pd.DataFrame(
            index=series_ssp[model][experiment].index
        )

    for area in areas:
        method = best_method_dict[(model, area)]
        best_method_used[model][area] = method

        cmip_hist = series_historical[model][area]
        obs = sst_obs[area]

        # -------------------------
        # Full period refit 1981-2014
        # -------------------------
        if method == 'linear':
            fitted_obj = fit_linear_full(
                cmip_hist,
                obs,
                calib_start='1981-01-01',
                calib_end='2014-12-31'
            )

            hist_best[model][area] = apply_linear_bc(cmip_hist, fitted_obj)

            for experiment in series_ssp[model]:
                ssp_best[model][experiment][area] = apply_linear_bc(
                    series_ssp[model][experiment][area],
                    fitted_obj
                )

        elif method == 'qm':
            fitted_obj = fit_qm_full(
                cmip_hist,
                obs,
                calib_start='1981-01-01',
                calib_end='2014-12-31'
            )

            hist_best[model][area] = apply_quantile_mapping(cmip_hist, fitted_obj)

            for experiment in series_ssp[model]:
                ssp_best[model][experiment][area] = apply_quantile_mapping(
                    series_ssp[model][experiment][area],
                    fitted_obj
                )

        elif method == 'xgb':
            fitted_obj = fit_xgb_full(
                cmip_hist,
                obs,
                calib_start='1981-01-01',
                calib_end='2014-12-31',
                random_state=42
            )

            hist_best[model][area] = apply_xgb_bc(cmip_hist, fitted_obj)

            for experiment in series_ssp[model]:
                ssp_best[model][experiment][area] = apply_xgb_bc(
                    series_ssp[model][experiment][area],
                    fitted_obj
                )

        else:
            raise ValueError(f"Método desconhecido: {method}")

        best_fitted_objects[model][area] = fitted_obj

    hist_best[model].index.name = 'time'

    for experiment in series_ssp[model]:
        ssp_best[model][experiment].index.name = 'time'


# ## Save the final results

# Historical

# In[ ]:


for model in models:
    hist_best[model].to_csv(
        os.path.join(SPLIT_PATH, f"{model}_historical_sst_best_method.csv")
    )


# SSPs

# In[ ]:


for model in models:
    for experiment in series_ssp[model]:
        ssp_best[model][experiment].to_csv(
            os.path.join(SPLIT_PATH, f"{model}_{experiment}_sst_best_method.csv")
        )


# Table with the chosen methods

# In[ ]:


rows = []
for model in models:
    for area in areas:
        rows.append({
            'model': model,
            'area': area,
            'selected_method': best_method_used[model][area]
        })

df_best_used = pd.DataFrame(rows)
df_best_used.to_csv(
    os.path.join(TABLE_SELECTED_METHODS_PATH, 'selected_best_method_applied.csv'),
    index=False
)


# # Step 3: figures

# In[84]:


method_label_map = {
    'linear': 'Linear Regression',
    'qm': 'Quantile Mapping',
    'xgb': 'XGBoost'
}

exp_label_map = {
    'ssp245': 'SSP2-4.5',
    'ssp585': 'SSP5-8.5'
}

test_start = '2005-01-01'
test_end   = '2014-12-31'

future_start = '2015-01-01'
future_end   = '2050-12-31'


# In[86]:


def monthly_climatology(series, start=None, end=None):
    s = series.copy()
    s.index = pd.to_datetime(s.index)

    if start is not None or end is not None:
        s = s.loc[start:end]

    s = pd.to_numeric(s, errors='coerce')
    clim = s.groupby(s.index.month).mean()
    return clim


# In[87]:


def empirical_cdf(series):
    s = pd.to_numeric(series, errors='coerce')
    s = s.dropna().sort_values()

    if len(s) == 0:
        return np.array([]), np.array([])

    x = s.to_numpy()
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


# In[88]:


def slice_series(series, start, end):
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = pd.to_numeric(s, errors='coerce')
    return s.loc[start:end]


# In[89]:


# ============================================================
# Area-specific axis limits for comparison figures
# Same limits within each area across all models
# ============================================================

def add_padding(vmin, vmax, pad_frac=0.05, min_pad=0.1):
    """
    Add proportional padding to axis limits.
    """
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None

    data_range = vmax - vmin
    pad = max(data_range * pad_frac, min_pad)
    return vmin - pad, vmax + pad


def get_numeric_values(series_list):
    """
    Concatenate numeric values from a list of pandas Series.
    """
    values = []

    for series in series_list:
        s = pd.to_numeric(series, errors='coerce').dropna()
        if len(s) > 0:
            values.extend(s.to_numpy())

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    return values


def make_bins_from_limits(limits, n_bins=20):
    """
    Create fixed histogram bins from predefined x-axis limits.
    """
    if limits is None:
        return None

    vmin, vmax = limits

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None

    if vmin == vmax:
        return np.linspace(vmin - 0.1, vmax + 0.1, n_bins)

    return np.linspace(vmin, vmax, n_bins)


def get_density_ylim(series_list, bins, pad_frac=0.10, min_pad=0.01):
    """
    Estimate a common y-axis limit for density histograms.
    """
    if bins is None:
        return None

    max_density = 0.0

    for series in series_list:
        s = pd.to_numeric(series, errors='coerce').dropna()

        if len(s) == 0:
            continue

        density, _ = np.histogram(s.to_numpy(), bins=bins, density=True)
        density = density[np.isfinite(density)]

        if len(density) > 0:
            max_density = max(max_density, density.max())

    if max_density <= 0:
        return None

    pad = max(max_density * pad_frac, min_pad)
    return 0, max_density + pad


# ------------------------------------------------------------
# Containers for limits by area
# ------------------------------------------------------------
test_ts_ylim_by_area = {}
test_clim_ylim_by_area = {}
test_dist_xlim_by_area = {}
test_dist_bins_by_area = {}
test_hist_ylim_by_area = {}

ssp_ts_ylim_by_area = {}
ssp_clim_ylim_by_area = {}
ssp_dist_xlim_by_area = {}
ssp_dist_bins_by_area = {}
ssp_hist_ylim_by_area = {}


for area in areas:
    # --------------------------------------------------------
    # Historical test period: time series limits
    # Original + corrected + observations, across all models
    # --------------------------------------------------------
    test_ts_series = []

    for model in models:
        test_ts_series.extend([
            slice_series(series_historical[model][area], test_start, test_end),
            slice_series(hist_best[model][area], test_start, test_end),
            slice_series(sst_obs[area], test_start, test_end)
        ])

    values = get_numeric_values(test_ts_series)
    test_ts_ylim_by_area[area] = add_padding(values.min(), values.max()) if len(values) > 0 else None

    # --------------------------------------------------------
    # Historical test period: monthly climatology limits
    # --------------------------------------------------------
    test_clim_series = []

    for model in models:
        test_clim_series.extend([
            monthly_climatology(series_historical[model][area], test_start, test_end),
            monthly_climatology(hist_best[model][area], test_start, test_end),
            monthly_climatology(sst_obs[area], test_start, test_end)
        ])

    values = get_numeric_values(test_clim_series)
    test_clim_ylim_by_area[area] = add_padding(values.min(), values.max()) if len(values) > 0 else None

    # --------------------------------------------------------
    # Historical test period: distribution/CDF limits and bins
    # --------------------------------------------------------
    test_dist_series = []

    for model in models:
        test_dist_series.extend([
            slice_series(series_historical[model][area], test_start, test_end),
            slice_series(hist_best[model][area], test_start, test_end),
            slice_series(sst_obs[area], test_start, test_end)
        ])

    values = get_numeric_values(test_dist_series)
    test_dist_xlim_by_area[area] = add_padding(values.min(), values.max()) if len(values) > 0 else None
    test_dist_bins_by_area[area] = make_bins_from_limits(test_dist_xlim_by_area[area], n_bins=20)
    test_hist_ylim_by_area[area] = get_density_ylim(test_dist_series, test_dist_bins_by_area[area])

    # --------------------------------------------------------
    # SSP period: time series limits
    # Original + corrected, across all models and experiments
    # --------------------------------------------------------
    ssp_ts_series = []

    for model in models:
        for experiment in series_ssp[model]:
            ssp_ts_series.extend([
                slice_series(series_ssp[model][experiment][area], future_start, future_end),
                slice_series(ssp_best[model][experiment][area], future_start, future_end)
            ])

    values = get_numeric_values(ssp_ts_series)
    ssp_ts_ylim_by_area[area] = add_padding(values.min(), values.max()) if len(values) > 0 else None

    # --------------------------------------------------------
    # SSP period: monthly climatology limits
    # --------------------------------------------------------
    ssp_clim_series = []

    for model in models:
        for experiment in series_ssp[model]:
            ssp_clim_series.extend([
                monthly_climatology(series_ssp[model][experiment][area], future_start, future_end),
                monthly_climatology(ssp_best[model][experiment][area], future_start, future_end)
            ])

    values = get_numeric_values(ssp_clim_series)
    ssp_clim_ylim_by_area[area] = add_padding(values.min(), values.max()) if len(values) > 0 else None

    # --------------------------------------------------------
    # SSP period: distribution/CDF limits and bins
    # --------------------------------------------------------
    ssp_dist_series = []

    for model in models:
        for experiment in series_ssp[model]:
            ssp_dist_series.extend([
                slice_series(series_ssp[model][experiment][area], future_start, future_end),
                slice_series(ssp_best[model][experiment][area], future_start, future_end)
            ])

    values = get_numeric_values(ssp_dist_series)
    ssp_dist_xlim_by_area[area] = add_padding(values.min(), values.max()) if len(values) > 0 else None
    ssp_dist_bins_by_area[area] = make_bins_from_limits(ssp_dist_xlim_by_area[area], n_bins=20)
    ssp_hist_ylim_by_area[area] = get_density_ylim(ssp_dist_series, ssp_dist_bins_by_area[area])


print('Area-specific limits defined for comparison figures:')
for area in areas:
    nome = area_display_name_map[area]
    area_file = area_file_name_map[area]
    print(f'\n{nome} ({area})')
    print(f'  Historical test time series ylim: {test_ts_ylim_by_area[area]}')
    print(f'  Historical test climatology ylim: {test_clim_ylim_by_area[area]}')
    print(f'  Historical test distribution xlim: {test_dist_xlim_by_area[area]}')
    print(f'  SSP time series ylim: {ssp_ts_ylim_by_area[area]}')
    print(f'  SSP climatology ylim: {ssp_clim_ylim_by_area[area]}')
    print(f'  SSP distribution xlim: {ssp_dist_xlim_by_area[area]}')


# ### Historical experiment

# Time series: original × corrected × observation

# In[ ]:


for model in models:
    for area in areas:
        nome = area_display_name_map[area]
        area_file = area_file_name_map[area]

        original = slice_series(series_historical[model][area], test_start, test_end)
        corrected = slice_series(hist_best[model][area], test_start, test_end)
        obs = slice_series(sst_obs[area], test_start, test_end)

        method_name = best_method_used[model][area]

        plt.figure(figsize=(12, 4))
        plt.plot(original.index, original, label='Original', alpha=0.7)
        plt.plot(corrected.index, corrected, label='Corrected', alpha=0.8)
        plt.plot(obs.index, obs, label='Observations', alpha=0.8, color='black')

        if test_ts_ylim_by_area[area] is not None:
            plt.ylim(test_ts_ylim_by_area[area])

        plt.legend(ncols=3)
        plt.grid(True)
        plt.title(f'{nome} - {model} - Test period ({test_start[:4]}–{test_end[:4]}) - {method_label_map[method_name]}')

        plt.savefig(
            os.path.join(FIGURE_TEST_TS_PATH, f'{model}_{area_file}_test_timeseries_best_method.png'),
            format='png', dpi=300, bbox_inches='tight'
        )
        plt.show()


# Monthly climatology

# In[ ]:


month_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

for model in models:
    for area in areas:
        nome = area_display_name_map[area]
        area_file = area_file_name_map[area]

        original = monthly_climatology(series_historical[model][area], test_start, test_end)
        corrected = monthly_climatology(hist_best[model][area], test_start, test_end)
        obs = monthly_climatology(sst_obs[area], test_start, test_end)

        method_name = best_method_used[model][area]

        plt.figure(figsize=(10, 4))
        plt.plot(original.index, original, marker='o', label='Original', alpha=0.7)
        plt.plot(corrected.index, corrected, marker='o', label='Corrected', alpha=0.8)
        plt.plot(obs.index, obs, marker='o', label='Observations', alpha=0.8, color='black')

        if test_clim_ylim_by_area[area] is not None:
            plt.ylim(test_clim_ylim_by_area[area])

        plt.xticks(range(1, 13), month_labels)
        plt.legend(ncols=3)
        plt.grid(True)
        plt.title(f'{nome} - {model} - Monthly climatology (test) - {method_label_map[method_name]}')

        plt.savefig(
            os.path.join(FIGURE_TEST_CLIM_PATH, f'{model}_{area_file}_test_climatology_best_method.png'),
            format='png', dpi=300, bbox_inches='tight'
        )
        plt.show()


# Distribuition + Cumulative Density Function

# In[ ]:


for model in models:
    for area in areas:
        nome = area_display_name_map[area]
        area_file = area_file_name_map[area]

        original = slice_series(series_historical[model][area], test_start, test_end).dropna()
        corrected = slice_series(hist_best[model][area], test_start, test_end).dropna()
        obs = slice_series(sst_obs[area], test_start, test_end).dropna()

        method_name = best_method_used[model][area]

        # Bins and limits are fixed by area across all models.
        bins = test_dist_bins_by_area[area]
        if bins is None:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histograma
        axes[0].hist(original, bins=bins, density=True, alpha=0.5, label='Original')
        axes[0].hist(corrected, bins=bins, density=True, alpha=0.5, label='Corrected')
        axes[0].hist(obs, bins=bins, density=True, color='black', alpha=0.5, label='Observations')
        axes[0].set_xlim(test_dist_xlim_by_area[area])
        if test_hist_ylim_by_area[area] is not None:
            axes[0].set_ylim(test_hist_ylim_by_area[area])
        axes[0].set_title('Distribution')
        axes[0].grid(True)
        axes[0].legend()

        # CDF
        x_o, y_o = empirical_cdf(original)
        x_c, y_c = empirical_cdf(corrected)
        x_obs, y_obs = empirical_cdf(obs)

        axes[1].plot(x_o, y_o, label='Original')
        axes[1].plot(x_c, y_c, label='Corrected')
        axes[1].plot(x_obs, y_obs, color='black', label='Observations')
        axes[1].set_xlim(test_dist_xlim_by_area[area])
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Empirical CDF')
        axes[1].grid(True)
        axes[1].legend()

        fig.suptitle(f'{nome} - {model} - Test distribution/CDF - {method_label_map[method_name]}')
        fig.tight_layout()

        plt.savefig(
            os.path.join(FIGURE_TEST_CDF_PATH, f'{model}_{area_file}_test_distribution_cdf_best_method.png'),
            format='png',
            dpi=300,
            bbox_inches='tight'
        )
        plt.show()


# ### SSPs

# Time series SSP: original × corrected

# In[ ]:


for model in models:
    for area in areas:
        nome = area_display_name_map[area]
        area_file = area_file_name_map[area]
        method_name = best_method_used[model][area]

        for experiment in series_ssp[model]:
            exp_name = exp_label_map.get(experiment, experiment)

            original = slice_series(series_ssp[model][experiment][area], future_start, future_end)
            corrected = slice_series(ssp_best[model][experiment][area], future_start, future_end)

            plt.figure(figsize=(12, 4))
            plt.plot(original.index, original, label='Original', alpha=0.7)
            plt.plot(corrected.index, corrected, label='Corrected', alpha=0.8)

            if ssp_ts_ylim_by_area[area] is not None:
                plt.ylim(ssp_ts_ylim_by_area[area])

            plt.legend(ncols=2)
            plt.grid(True)
            plt.title(f'{nome} - {model} - {exp_name} ({future_start[:4]}–{future_end[:4]}) - {method_label_map[method_name]}')

            plt.savefig(
                os.path.join(FIGURE_SSP_TS_PATH, f'{model}_{experiment}_{area_file}_future_timeseries_best_method.png'),
                format='png', dpi=300, bbox_inches='tight'
            )
            plt.show()


# Monthly climatology

# In[ ]:


for model in models:
    for area in areas:
        nome = area_display_name_map[area]
        area_file = area_file_name_map[area]
        method_name = best_method_used[model][area]

        for experiment in series_ssp[model]:
            exp_name = exp_label_map.get(experiment, experiment)

            original = monthly_climatology(series_ssp[model][experiment][area], future_start, future_end)
            corrected = monthly_climatology(ssp_best[model][experiment][area], future_start, future_end)

            plt.figure(figsize=(10, 4))
            plt.plot(original.index, original, marker='o', label='Original', alpha=0.7)
            plt.plot(corrected.index, corrected, marker='o', label='Corrected', alpha=0.8)

            if ssp_clim_ylim_by_area[area] is not None:
                plt.ylim(ssp_clim_ylim_by_area[area])

            plt.xticks(range(1, 13), month_labels)
            plt.legend(ncols=2)
            plt.grid(True)
            plt.title(f'{nome} - {model} - {exp_name} climatology ({future_start[:4]}–{future_end[:4]}) - {method_label_map[method_name]}')

            plt.savefig(
                os.path.join(FIGURE_SSP_CLIM_PATH, f'{model}_{experiment}_{area_file}_future_climatology_best_method.png'),
                format='png', dpi=300, bbox_inches='tight'
            )
            plt.show()


# Distribution + Cumulative Density Function SSPs

# In[ ]:


for model in models:
    for area in areas:
        nome = area_display_name_map[area]
        area_file = area_file_name_map[area]
        method_name = best_method_used[model][area]

        for experiment in series_ssp[model]:
            exp_name = exp_label_map.get(experiment, experiment)

            original = slice_series(series_ssp[model][experiment][area], future_start, future_end).dropna()
            corrected = slice_series(ssp_best[model][experiment][area], future_start, future_end).dropna()

            # Bins and limits are fixed by area across all models and SSP experiments.
            bins = ssp_dist_bins_by_area[area]
            if bins is None:
                continue

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Histograma
            axes[0].hist(original, bins=bins, density=True, alpha=0.5, label='Original')
            axes[0].hist(corrected, bins=bins, density=True, alpha=0.5, label='Corrected')
            axes[0].set_xlim(ssp_dist_xlim_by_area[area])
            if ssp_hist_ylim_by_area[area] is not None:
                axes[0].set_ylim(ssp_hist_ylim_by_area[area])
            axes[0].set_title('Distribution')
            axes[0].grid(True)
            axes[0].legend()

            # CDF
            x_o, y_o = empirical_cdf(original)
            x_c, y_c = empirical_cdf(corrected)

            axes[1].plot(x_o, y_o, label='Original')
            axes[1].plot(x_c, y_c, label='Corrected')
            axes[1].set_xlim(ssp_dist_xlim_by_area[area])
            axes[1].set_ylim(0, 1)
            axes[1].set_title('Empirical CDF')
            axes[1].grid(True)
            axes[1].legend()

            fig.suptitle(f'{nome} - {model} - {exp_name} distribution/CDF ({future_start[:4]}–{future_end[:4]}) - {method_label_map[method_name]}')
            fig.tight_layout()

            plt.savefig(
                os.path.join(FIGURE_SSP_CDF_PATH, f'{model}_{experiment}_{area_file}_future_distribution_cdf_best_method.png'),
                format='png',
                dpi=300,
                bbox_inches='tight'
            )
            plt.show()


# ### Summary figures  

# Heatmap of the chosen method by model and area

# In[98]:


method_code_map = {
    'linear': 0,
    'qm': 1,
    'xgb': 2
}

method_label_map = {
    0: 'Linear',
    1: 'QM',
    2: 'XGBoost'
}

method_label_map2 = {
    'linear': 'Linear',
    'qm': 'QM',
    'xgb': 'XGBoost'
}


# In[99]:


heatmap_df = best_methods_robust.pivot(
    index='area',
    columns='model',
    values='method'
)

area_label_map = area_display_name_map.copy()

heatmap_df = heatmap_df.rename(index=area_label_map)

heatmap_num = heatmap_df.replace(method_code_map)


# In[ ]:


# 3 cores discretas, uma para cada método
cmap = ListedColormap(['#4C72B0', '#55A868', '#C44E52'])
norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5], ncolors=cmap.N)

fig, ax = plt.subplots(figsize=(max(6, len(heatmap_num.columns) * 0.55),
                                max(3.5, len(heatmap_num.index) * 0.42)))

im = ax.imshow(heatmap_num.values, cmap=cmap, norm=norm, aspect='auto')

ax.set_xticks(np.arange(len(heatmap_num.columns)))
ax.set_yticks(np.arange(len(heatmap_num.index)))

ax.set_xticklabels(heatmap_num.columns, rotation=45, ha='right')
ax.set_yticklabels(heatmap_num.index)

# Texto nas células
for i in range(len(heatmap_num.index)):
    for j in range(len(heatmap_num.columns)):
        method_name = heatmap_df.iloc[i, j]
        label = {
            'linear': 'Linear',
            'qm': 'QM',
            'xgb': 'XGB'
        }.get(method_name, method_name)
        ax.text(j, i, label, ha='center', va='center', fontsize=8)

# Legenda categórica
# legend_elements = [
#     Patch(facecolor='#4C72B0', edgecolor='none', label='Linear'),
#     Patch(facecolor='#55A868', edgecolor='none', label='QM'),
#     Patch(facecolor='#C44E52', edgecolor='none', label='XGBoost')
# ]

# ax.legend(
#     handles=legend_elements,
#     loc='upper center',
#     bbox_to_anchor=(0.5, -0.12),
#     ncol=3,
#     frameon=False
# )

ax.set_title('Selected bias-correction method by model and area')
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, 'heatmap_selected_method_by_model_area_discrete.png'),
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# Barplot with method frequency

# In[141]:


method_counts = best_methods_robust['method'].value_counts().reindex(['linear', 'qm', 'xgb'], fill_value=0)
method_freq = 100 * method_counts / method_counts.sum()
print(method_freq)


# In[103]:


method_label_map2


# In[ ]:


plt.figure(figsize=(6, 4))
bars = plt.bar([method_label_map2.get(method, method) for method in method_freq.index], method_freq.values, color = ['#4C72B0', '#55A868', '#C44E52'])

for bar, val in zip(bars, method_freq.values):
    plt.text(bar.get_x() + bar.get_width()/2, val, f'{val:.1f}%', ha='center', va='bottom')

plt.ylabel('Frequency (%)')
plt.title('Frequency of selected bias-correction methods')
plt.grid(axis='y')

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, 'barplot_method_selection_frequency.png'),
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# RMSE boxplot by method

# In[ ]:


rmse_data = [
    df_metrics_all.loc[df_metrics_all['method'] == method, 'rmse'].dropna().values
    for method in ['linear', 'qm', 'xgb']
]

plt.figure(figsize=(7, 4))
plt.boxplot(rmse_data, labels=['Linear', 'QM', 'XGBoost'])
plt.ylabel('RMSE')
plt.title('RMSE distribution by method')
plt.grid(True)

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, 'boxplot_rmse_by_method.png'),
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# MAE boxplot by method

# In[ ]:


mae_data = [
    df_metrics_all.loc[df_metrics_all['method'] == method, 'mae'].dropna().values
    for method in ['linear', 'qm', 'xgb']
]

plt.figure(figsize=(7, 4))
plt.boxplot(mae_data, labels=['Linear', 'QM', 'XGBoost'])
plt.ylabel('MAE')
plt.title('MAE distribution by method')
plt.grid(True)

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, 'boxplot_mae_by_method.png'),
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# RMSE and MAE boxplot by method

# In[ ]:


def annotate_boxplot_medians(ax, boxplot_dict, fmt='{:.2f}', y_offset_frac=0.02):
    """
    Annotate the median value of each boxplot directly above the median line.
    """
    y_min, y_max = ax.get_ylim()
    y_offset = (y_max - y_min) * y_offset_frac

    for i, median_line in enumerate(boxplot_dict['medians'], start=1):
        median_value = median_line.get_ydata()[0]
        ax.text(
            i,
            median_value + y_offset,
            fmt.format(median_value),
            ha='center',
            va='bottom',
            fontsize=9
        )


def metric_ylim_with_baseline(data_list, baseline_value, pad_frac=0.10, min_pad=0.05):
    """
    Define y-axis limits considering the boxplot values and the baseline line.
    """
    values = []

    for data in data_list:
        values.extend(np.asarray(data, dtype=float)[~np.isnan(data)])

    if not np.isnan(baseline_value):
        values.append(baseline_value)

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return None

    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    data_range = vmax - vmin
    pad = max(data_range * pad_frac, min_pad)

    lower = max(0, vmin - pad)
    upper = vmax + pad

    return lower, upper


# Same y-axis range for RMSE and MAE to facilitate direct visual comparison.
rmse_ylim = metric_ylim_with_baseline(rmse_data, baseline_rmse_median)
mae_ylim = metric_ylim_with_baseline(mae_data, baseline_mae_median)

combined_lower = min(rmse_ylim[0], mae_ylim[0])
combined_upper = max(rmse_ylim[1], mae_ylim[1])
shared_metric_ylim = (combined_lower, combined_upper)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

labels = ['Linear', 'QM', 'XGBoost']

# ------------------------------------------------------------
# RMSE
# ------------------------------------------------------------
bp_rmse = axes[0].boxplot(rmse_data, labels=labels)

axes[0].axhline(
    baseline_rmse_median,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=f'Raw historical median = {baseline_rmse_median:.2f}'
)

axes[0].set_ylim(shared_metric_ylim)
annotate_boxplot_medians(axes[0], bp_rmse)

axes[0].set_ylabel('Error (°C)')
axes[0].set_title('RMSE by method')
axes[0].grid(True)
axes[0].legend(fontsize=8)

# ------------------------------------------------------------
# MAE
# ------------------------------------------------------------
bp_mae = axes[1].boxplot(mae_data, labels=labels)

axes[1].axhline(
    baseline_mae_median,
    color='red',
    linestyle='--',
    linewidth=1.5,
    label=f'Raw historical median = {baseline_mae_median:.2f}'
)

axes[1].set_ylim(shared_metric_ylim)
annotate_boxplot_medians(axes[1], bp_mae)

axes[1].set_ylabel('Error (°C)')
axes[1].set_title('MAE by method')
axes[1].grid(True)
axes[1].legend(fontsize=8)

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, 'boxplot_rmse_mae_by_method.png'),
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# In[ ]:


# ============================================================
# Boxplot: RMSE distribution for raw historical and correction methods
# ============================================================

rmse_raw_data = df_metrics_baseline["rmse"].dropna().values

rmse_method_data = [
    df_metrics_all.loc[df_metrics_all["method"] == method, "rmse"].dropna().values
    for method in ["linear", "qm", "xgb"]
]

rmse_data_with_raw = [
    rmse_raw_data,
    *rmse_method_data
]

labels_rmse_with_raw = [
    "Raw",
    "Linear",
    "QM",
    "XGBoost"
]

fig, ax = plt.subplots(figsize=(8, 4.5))

bp_rmse_raw = ax.boxplot(
    rmse_data_with_raw,
    labels=labels_rmse_with_raw
)

annotate_boxplot_medians(ax, bp_rmse_raw)

ax.set_ylabel("RMSE (°C)")
ax.set_xlabel("Dataset / bias-correction method")
ax.set_title("RMSE distribution before and after bias correction")
ax.grid(True, axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, "boxplot_rmse_raw_linear_qm_xgb.png"),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Barplot by area counting wins by method

# In[108]:


wins_by_area = (
    best_methods_robust
    .groupby(['area', 'method'])
    .size()
    .unstack(fill_value=0)
)

color_map_methods = {
    'linear': '#4C72B0',
    'qm': '#55A868',
    'xgb': '#C44E52'
}


# In[109]:


wins_by_area = wins_by_area.reindex(columns=['linear', 'qm', 'xgb'], fill_value=0)


# In[110]:


# Exemplo, use só se precisar
wins_by_area = wins_by_area.rename(index=area_label_map)


# In[ ]:


# ax = wins_by_area.plot(
#     kind='bar',
#     figsize=(10, 4.5),
#     width=0.8
# )

ax = wins_by_area.plot(
    kind='bar',
    figsize=(10, 4.5),
    width=0.8,
    color=[color_map_methods[m] for m in wins_by_area.columns]
)

plt.ylabel('Number of selections')
plt.xlabel('Area')
plt.title('Method selection frequency by area')
plt.grid(axis='y')
plt.legend(['Linear', 'QM', 'XGBoost'], title='Method', ncols=3)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, 'barplot_method_selection_frequency_by_area.png'),
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# Stacked barplot

# In[ ]:


# ax = wins_by_area.plot(
#     kind='bar',
#     stacked=True,
#     figsize=(10, 4.5),
#     width=0.8
# )

ax = wins_by_area.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 4.5),
    width=0.8,
    color=[color_map_methods[m] for m in wins_by_area.columns]
)

plt.ylabel('Number of selections')
plt.xlabel('Area')
plt.title('Method selection frequency by area')
plt.grid(axis='y')
plt.legend(['Linear', 'QM', 'XGBoost'], title='Method', ncols=3)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, 'stacked_barplot_method_selection_frequency_by_area.png'),
    format='png',
    dpi=300,
    bbox_inches='tight'
)
plt.show()


# ### Calculation of bias improvement - done with the test (2005-2014) because it is independent of calibration.

# In[113]:


# ============================================================
# Mean SST bias-correction improvement relative to observations
# ============================================================

def compute_correction_improvement(original, corrected, obs):
    """
    Compute SST bias-correction improvement relative to observations.

    Improvement is defined as:

        |original - obs| - |corrected - obs|

    Positive values mean the correction reduced the absolute error.
    """

    original = original.copy()
    corrected = corrected.copy()
    obs = obs.copy()

    original.index = pd.to_datetime(original.index).to_period("M")
    corrected.index = pd.to_datetime(corrected.index).to_period("M")
    obs.index = pd.to_datetime(obs.index).to_period("M")

    original = pd.to_numeric(original, errors="coerce")
    corrected = pd.to_numeric(corrected, errors="coerce")
    obs = pd.to_numeric(obs, errors="coerce")

    df = pd.concat(
        [
            original.rename("original"),
            corrected.rename("corrected"),
            obs.rename("obs")
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(df) == 0:
        return {
            "n": 0,
            "mean_abs_error_original": np.nan,
            "mean_abs_error_corrected": np.nan,
            "mean_improvement_abs_error": np.nan,
            "median_improvement_abs_error": np.nan,
            "min_improvement_abs_error": np.nan,
            "max_improvement_abs_error": np.nan,
            "mean_signed_bias_original": np.nan,
            "mean_signed_bias_corrected": np.nan,
            "mean_abs_correction_applied": np.nan,
            "mean_signed_correction_applied": np.nan,
            "relative_mae_reduction_percent": np.nan
        }

    abs_error_original = (df["original"] - df["obs"]).abs()
    abs_error_corrected = (df["corrected"] - df["obs"]).abs()

    improvement = abs_error_original - abs_error_corrected

    signed_bias_original = df["original"] - df["obs"]
    signed_bias_corrected = df["corrected"] - df["obs"]

    correction_applied = df["corrected"] - df["original"]

    mean_abs_error_original = abs_error_original.mean()
    mean_abs_error_corrected = abs_error_corrected.mean()

    relative_mae_reduction_percent = (
        (mean_abs_error_original - mean_abs_error_corrected)
        / mean_abs_error_original
        * 100
        if mean_abs_error_original != 0
        else np.nan
    )

    return {
        "n": len(df),
        "mean_abs_error_original": mean_abs_error_original,
        "mean_abs_error_corrected": mean_abs_error_corrected,
        "mean_improvement_abs_error": improvement.mean(),
        "median_improvement_abs_error": improvement.median(),
        "min_improvement_abs_error": improvement.min(),
        "max_improvement_abs_error": improvement.max(),
        "mean_signed_bias_original": signed_bias_original.mean(),
        "mean_signed_bias_corrected": signed_bias_corrected.mean(),
        "mean_abs_correction_applied": correction_applied.abs().mean(),
        "mean_signed_correction_applied": correction_applied.mean(),
        "relative_mae_reduction_percent": relative_mae_reduction_percent
    }


# In[ ]:


# ============================================================
# Improvement during the independent test period
# ============================================================

improvement_rows = []

for model in models:
    for area in areas:

        original = slice_series(
            series_historical[model][area],
            test_start,
            test_end
        )

        corrected = slice_series(
            hist_best[model][area],
            test_start,
            test_end
        )

        obs = slice_series(
            sst_obs[area],
            test_start,
            test_end
        )

        metrics = compute_correction_improvement(
            original=original,
            corrected=corrected,
            obs=obs
        )

        row = {
            "model": model,
            "area": area,
            "selected_method": best_method_used[model][area],
            "period_start": test_start,
            "period_end": test_end
        }

        row.update(metrics)
        improvement_rows.append(row)

df_improvement_test = pd.DataFrame(improvement_rows)

df_improvement_test.to_csv(
    os.path.join(TABLE_IMPROVEMENT_PATH, "sst_bias_correction_improvement_test_period.csv"),
    index=False
)

df_improvement_test


# In[ ]:


# ============================================================
# Area-level summary across models
# ============================================================

df_improvement_by_area = (
    df_improvement_test
    .groupby("area", as_index=False)
    .agg(
        n_model_area=("model", "count"),
        mean_improvement_abs_error=("mean_improvement_abs_error", "mean"),
        min_improvement_abs_error=("mean_improvement_abs_error", "min"),
        max_improvement_abs_error=("mean_improvement_abs_error", "max"),
        mean_abs_error_original=("mean_abs_error_original", "mean"),
        mean_abs_error_corrected=("mean_abs_error_corrected", "mean"),
        mean_relative_mae_reduction_percent=("relative_mae_reduction_percent", "mean"),
        mean_abs_correction_applied=("mean_abs_correction_applied", "mean")
    )
)

df_improvement_by_area.to_csv(
    os.path.join(TABLE_IMPROVEMENT_PATH, "sst_bias_correction_improvement_by_area.csv"),
    index=False
)

df_improvement_by_area


# In[ ]:


# ============================================================
# Overall summary across all models and areas
# ============================================================

df_improvement_overall = pd.DataFrame({
    "metric": [
        "mean_improvement_abs_error",
        "mean_abs_error_original",
        "mean_abs_error_corrected",
        "relative_mae_reduction_percent",
        "mean_abs_correction_applied"
    ],
    "minimum": [
        df_improvement_test["mean_improvement_abs_error"].min(),
        df_improvement_test["mean_abs_error_original"].min(),
        df_improvement_test["mean_abs_error_corrected"].min(),
        df_improvement_test["relative_mae_reduction_percent"].min(),
        df_improvement_test["mean_abs_correction_applied"].min()
    ],
    "mean": [
        df_improvement_test["mean_improvement_abs_error"].mean(),
        df_improvement_test["mean_abs_error_original"].mean(),
        df_improvement_test["mean_abs_error_corrected"].mean(),
        df_improvement_test["relative_mae_reduction_percent"].mean(),
        df_improvement_test["mean_abs_correction_applied"].mean()
    ],
    "maximum": [
        df_improvement_test["mean_improvement_abs_error"].max(),
        df_improvement_test["mean_abs_error_original"].max(),
        df_improvement_test["mean_abs_error_corrected"].max(),
        df_improvement_test["relative_mae_reduction_percent"].max(),
        df_improvement_test["mean_abs_correction_applied"].max()
    ]
})

df_improvement_overall.to_csv(
    os.path.join(TABLE_IMPROVEMENT_PATH, "sst_bias_correction_improvement_overall_summary.csv"),
    index=False
)

df_improvement_overall


# The average improvement was estimated as the reduction in absolute error between the simulated historical series and the observation, before and after bias correction. Positive values ​​indicate that the corrected series is closer to the observation, while negative values ​​indicate local degradation of performance.

# In[ ]:


# ============================================================
# Barplot: mean improvement by area
# ============================================================

# Keep the original area names for data handling, but use display names
# only for the x-axis labels.
df_plot = df_improvement_by_area.copy()
df_plot["area_label"] = df_plot["area"].map(area_display_name_map)

fig, ax = plt.subplots(figsize=(10, 4))

x = np.arange(len(df_plot))

ax.bar(
    x,
    df_plot["mean_improvement_abs_error"]
)

ax.axhline(0, color="black", linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(
    df_plot["area_label"],
    rotation=45,
    ha="right"
)

ax.set_ylabel("Mean reduction in absolute error (°C)")
ax.set_xlabel("Area")
ax.set_title("Mean SST bias-correction improvement by area")
ax.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, "barplot_mean_sst_improvement_by_area.png"),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[ ]:


# ============================================================
# Heatmap: mean improvement by model and area
# ============================================================

heatmap_improvement = df_improvement_test.pivot(
    index="model",
    columns="area",
    values="mean_improvement_abs_error"
)

heatmap_improvement = heatmap_improvement.rename(columns=area_label_map)

fig, ax = plt.subplots(figsize=(10, 4))

im = ax.imshow(heatmap_improvement.values, aspect="auto")

ax.set_xticks(np.arange(len(heatmap_improvement.columns)))
ax.set_yticks(np.arange(len(heatmap_improvement.index)))

ax.set_xticklabels(heatmap_improvement.columns, rotation=45, ha="right")
ax.set_yticklabels(heatmap_improvement.index)

for i in range(len(heatmap_improvement.index)):
    for j in range(len(heatmap_improvement.columns)):
        value = heatmap_improvement.iloc[i, j]
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Mean reduction in absolute error (°C)")

ax.set_title("Mean SST bias-correction improvement by model and area")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, "heatmap_mean_sst_improvement_model_area.png"),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# In[119]:


df_improvement_test[[
    "model",
    "area",
    "selected_method",
    "n",
    "mean_abs_error_original",
    "mean_abs_error_corrected",
    "mean_improvement_abs_error",
    "relative_mae_reduction_percent"
]].sort_values("mean_improvement_abs_error", ascending=False)


# Check if there were any cases where the correction worsened.

# In[120]:


df_improvement_test[
    df_improvement_test["mean_improvement_abs_error"] < 0
][[
    "model",
    "area",
    "selected_method",
    "mean_improvement_abs_error",
    "relative_mae_reduction_percent"
]]


# Overall summary

# In[121]:


df_improvement_test["mean_improvement_abs_error"].describe()


# In[122]:


min_imp = df_improvement_test["mean_improvement_abs_error"].min()
mean_imp = df_improvement_test["mean_improvement_abs_error"].mean()
max_imp = df_improvement_test["mean_improvement_abs_error"].max()

print(f"Minimum improvement: {min_imp:.2f} °C")
print(f"Mean improvement: {mean_imp:.2f} °C")
print(f"Maximum improvement: {max_imp:.2f} °C")


# ### Assess which model/area needed the most correction (original - note)

# In[123]:


# ============================================================
# Helper: get best corrected prediction in the independent test period
# ============================================================

def get_best_test_prediction(model, area):
    method = best_method_used[model][area]

    if method == "linear":
        return linear_test_pred[model][area]
    elif method == "qm":
        return qm_test_pred[model][area]
    elif method == "xgb":
        return xgb_test_pred[model][area]
    else:
        raise ValueError(f"Unknown method: {method}")


# In[124]:


# ============================================================
# Helpers
# ============================================================

def force_monthly_period_index(series):
    """
    Convert a Series index to monthly PeriodIndex safely.

    Handles DatetimeIndex, PeriodIndex, strings, and date-like indexes.
    """

    series = series.copy()

    if isinstance(series.index, pd.PeriodIndex):
        series.index = series.index.asfreq("M")

    else:
        series.index = pd.to_datetime(series.index).to_period("M")

    return series


# ============================================================
# Metrics to quantify how much each model-area needs correction
# ============================================================

def compute_correction_need_metrics(original, corrected, obs):
    """
    Compute metrics describing how much correction is needed
    and how much correction was applied.

    Positive signed bias means the model is warmer than observations.
    Negative signed bias means the model is colder than observations.
    """

    original = force_monthly_period_index(original)
    corrected = force_monthly_period_index(corrected)
    obs = force_monthly_period_index(obs)

    original = pd.to_numeric(original, errors="coerce")
    corrected = pd.to_numeric(corrected, errors="coerce")
    obs = pd.to_numeric(obs, errors="coerce")

    df = pd.concat(
        [
            original.rename("original"),
            corrected.rename("corrected"),
            obs.rename("obs"),
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(df) == 0:
        return {
            "n": 0,
            "mae_original": np.nan,
            "rmse_original": np.nan,
            "mean_signed_bias_original": np.nan,
            "mean_abs_bias_original": np.nan,
            "mae_corrected": np.nan,
            "mean_abs_correction_applied": np.nan,
            "mean_signed_correction_applied": np.nan,
            "improvement_abs_error": np.nan
        }

    err_original = df["original"] - df["obs"]
    err_corrected = df["corrected"] - df["obs"]
    correction_applied = df["corrected"] - df["original"]

    mae_original = np.mean(np.abs(err_original))
    rmse_original = np.sqrt(np.mean(err_original ** 2))
    mean_signed_bias_original = np.mean(err_original)
    mean_abs_bias_original = np.mean(np.abs(err_original))

    mae_corrected = np.mean(np.abs(err_corrected))

    mean_abs_correction_applied = np.mean(np.abs(correction_applied))
    mean_signed_correction_applied = np.mean(correction_applied)

    improvement_abs_error = mae_original - mae_corrected

    return {
        "n": len(df),
        "mae_original": mae_original,
        "rmse_original": rmse_original,
        "mean_signed_bias_original": mean_signed_bias_original,
        "mean_abs_bias_original": mean_abs_bias_original,
        "mae_corrected": mae_corrected,
        "mean_abs_correction_applied": mean_abs_correction_applied,
        "mean_signed_correction_applied": mean_signed_correction_applied,
        "improvement_abs_error": improvement_abs_error
    }


# In[ ]:


# ============================================================
# Build correction-need summary table
# ============================================================

rows_need = []

for model in models:
    for area in areas:
        original = slice_series(
            series_historical[model][area],
            test_start,
            test_end
        )

        corrected = get_best_test_prediction(model, area)

        obs = slice_series(
            sst_obs[area],
            test_start,
            test_end
        )

        metrics = compute_correction_need_metrics(
            original=original,
            corrected=corrected,
            obs=obs
        )

        row = {
            "model": model,
            "area": area,
            "area_name": area_display_name_map[area],
            "selected_method": best_method_used[model][area]
        }
        row.update(metrics)
        rows_need.append(row)

df_correction_need = pd.DataFrame(rows_need)

df_correction_need.to_csv(
    os.path.join(TABLE_METRICS_PATH, "sst_correction_need_summary_test_period.csv"),
    index=False
)

df_correction_need


# Heatmap of correction needs (original MAE)

# The original mean absolute error highlights the model–area combinations that required the largest SST bias correction. Larger values indicate regions where the raw CMIP6 SST indices were farther from observations and therefore required stronger correction efforts.

# In[ ]:


# ============================================================
# Heatmap 1: original MAE (how much each model-area needs correction)
# ============================================================

area_name_map = area_display_name_map.copy()

heatmap_mae_need = (
    df_correction_need
    .pivot(index="model", columns="area", values="mae_original")
    .reindex(index=models, columns=areas)
)

fig, ax = plt.subplots(figsize=(14, 4.5))

im = ax.imshow(heatmap_mae_need.values, aspect="auto")

ax.set_xticks(np.arange(len(areas)))
ax.set_yticks(np.arange(len(models)))

ax.set_xticklabels([area_name_map[a] for a in areas], rotation=45, ha="right")
ax.set_yticklabels(models)

for i in range(len(models)):
    for j in range(len(areas)):
        value = heatmap_mae_need.iloc[i, j]
        if pd.notna(value):
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=8
            )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Original MAE relative to observations (°C)")

ax.set_title("How much each model–area needs correction\nOriginal SST error in the independent test period")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, "heatmap_correction_need_mae_original.png"),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Heatmap of the original signed bias

# The signed bias indicates whether the original SST was systematically overestimated or underestimated. Positive values indicate warm bias, whereas negative values indicate cold bias.

# In[ ]:


# ============================================================
# Heatmap 2: signed original bias (direction of required correction)
# ============================================================

heatmap_signed_bias = (
    df_correction_need
    .pivot(index="model", columns="area", values="mean_signed_bias_original")
    .reindex(index=models, columns=areas)
)

vmax = np.nanmax(np.abs(heatmap_signed_bias.values))
vmin = -vmax

fig, ax = plt.subplots(figsize=(14, 4.5))

im = ax.imshow(
    heatmap_signed_bias.values,
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
    cmap="coolwarm"
)

ax.set_xticks(np.arange(len(areas)))
ax.set_yticks(np.arange(len(models)))

ax.set_xticklabels([area_name_map[a] for a in areas], rotation=45, ha="right")
ax.set_yticklabels(models)

for i in range(len(models)):
    for j in range(len(areas)):
        value = heatmap_signed_bias.iloc[i, j]
        if pd.notna(value):
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=8
            )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Original signed bias (°C)")

ax.set_title("Direction of correction needed\nPositive = model too warm, negative = model too cold")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, "heatmap_correction_need_signed_bias_original.png"),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Heatmap of the correction actually applied

# The magnitude of the applied correction shows how strongly the selected bias-correction method modified each model–area SST series. This complements the original-error heatmap by distinguishing the initial need for correction from the actual amplitude of the statistical adjustment.

# In[ ]:


# ============================================================
# Heatmap 3: magnitude of correction applied
# ============================================================

heatmap_applied = (
    df_correction_need
    .pivot(index="model", columns="area", values="mean_abs_correction_applied")
    .reindex(index=models, columns=areas)
)

fig, ax = plt.subplots(figsize=(14, 4.5))

im = ax.imshow(heatmap_applied.values, aspect="auto")

ax.set_xticks(np.arange(len(areas)))
ax.set_yticks(np.arange(len(models)))

ax.set_xticklabels([area_name_map[a] for a in areas], rotation=45, ha="right")
ax.set_yticklabels(models)

for i in range(len(models)):
    for j in range(len(areas)):
        value = heatmap_applied.iloc[i, j]
        if pd.notna(value):
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                fontsize=8
            )

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Mean absolute correction applied (°C)")

ax.set_title("Magnitude of SST correction applied in the independent test period")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, "heatmap_correction_applied_mean_abs.png"),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Ranking of corrections by area

# In[ ]:


# ============================================================
# Barplot: average correction need by area
# ============================================================

df_need_by_area = (
    df_correction_need
    .groupby(["area", "area_name"], as_index=False)
    .agg(
        mean_mae_original=("mae_original", "mean"),
        min_mae_original=("mae_original", "min"),
        max_mae_original=("mae_original", "max"),
        mean_abs_correction_applied=("mean_abs_correction_applied", "mean")
    )
    .sort_values("mean_mae_original", ascending=False)
)

plt.figure(figsize=(10, 5))

plt.bar(
    df_need_by_area["area_name"],
    df_need_by_area["mean_mae_original"]
)

plt.ylabel("Mean original MAE (°C)")
plt.xlabel("Area")
plt.title("Average SST correction need by area")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, "barplot_average_correction_need_by_area.png"),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ### Quantitative assessment of how the error improved:
# 
# - Original already similar to observations
# - - The original data already had a low error rate.
# 
# - Correction successful / challenging but improved
# - - The original data had a significant error rate, but the correction reduced the error rate.
# 
# - Correction failed / no clear correction pattern
# - - The original data had a significant error rate, and the correction did not improve it sufficiently, or it worsened it.

# In[130]:


# ============================================================
# Automatic classification of climatological correction behavior
# ============================================================

def classify_climatology_correction(
    mae_original,
    mae_corrected,
    close_threshold=0.50,
    improvement_threshold_percent=70.0
):
    """
    Classify correction behavior based on monthly climatological MAE.

    Parameters
    ----------
    mae_original : float
        MAE between original model climatology and observed climatology.
    mae_corrected : float
        MAE between corrected model climatology and observed climatology.
    close_threshold : float
        Threshold in °C below which the original data is considered already close
        to observations.
    improvement_threshold_percent : float
        Minimum relative MAE reduction required to classify correction as successful.

    Returns
    -------
    str
        Correction behavior class.
    """

    if not np.isfinite(mae_original) or not np.isfinite(mae_corrected):
        return "Invalid or missing data"

    if mae_original <= close_threshold:
        return "Original close to observations"

    if mae_original == 0:
        return "Original close to observations"

    relative_improvement = (
        (mae_original - mae_corrected) / mae_original * 100
    )

    if relative_improvement >= improvement_threshold_percent:
        return "Correction successful"

    return "Correction limited or failed"


def compute_climatology_correction_metrics(original, corrected, obs):
    """
    Compute correction metrics using monthly climatologies.
    """

    clim_original = monthly_climatology(original, test_start, test_end)
    clim_corrected = monthly_climatology(corrected, test_start, test_end)
    clim_obs = monthly_climatology(obs, test_start, test_end)

    df = pd.concat(
        [
            clim_original.rename("original"),
            clim_corrected.rename("corrected"),
            clim_obs.rename("obs")
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(df) == 0:
        return {
            "n_months": 0,
            "mae_original_clim": np.nan,
            "mae_corrected_clim": np.nan,
            "rmse_original_clim": np.nan,
            "rmse_corrected_clim": np.nan,
            "improvement_clim": np.nan,
            "relative_improvement_clim_percent": np.nan
        }

    error_original = df["original"] - df["obs"]
    error_corrected = df["corrected"] - df["obs"]

    mae_original = np.mean(np.abs(error_original))
    mae_corrected = np.mean(np.abs(error_corrected))

    rmse_original = np.sqrt(np.mean(error_original ** 2))
    rmse_corrected = np.sqrt(np.mean(error_corrected ** 2))

    improvement = mae_original - mae_corrected

    if mae_original == 0:
        relative_improvement = np.nan
    else:
        relative_improvement = improvement / mae_original * 100

    return {
        "n_months": len(df),
        "mae_original_clim": mae_original,
        "mae_corrected_clim": mae_corrected,
        "rmse_original_clim": rmse_original,
        "rmse_corrected_clim": rmse_corrected,
        "improvement_clim": improvement,
        "relative_improvement_clim_percent": relative_improvement
    }


# In[ ]:


# ============================================================
# Build climatological correction classification table
# ============================================================

area_name_map = area_display_name_map.copy()

classification_rows = []

for model in models:
    for area in areas:

        original = slice_series(
            series_historical[model][area],
            test_start,
            test_end
        )

        corrected = slice_series(
            hist_best[model][area],
            test_start,
            test_end
        )

        obs = slice_series(
            sst_obs[area],
            test_start,
            test_end
        )

        metrics = compute_climatology_correction_metrics(
            original=original,
            corrected=corrected,
            obs=obs
        )

        correction_class = classify_climatology_correction(
            mae_original=metrics["mae_original_clim"],
            mae_corrected=metrics["mae_corrected_clim"],
            close_threshold=0.30,
            improvement_threshold_percent=30.0
        )

        row = {
            "model": model,
            "area": area,
            "area_name": area_name_map[area],
            "selected_method": best_method_used[model][area],
            "correction_class": correction_class
        }

        row.update(metrics)
        classification_rows.append(row)

df_clim_classification = pd.DataFrame(classification_rows)

df_clim_classification.to_csv(
    os.path.join(TABLE_METRICS_PATH, "sst_climatology_correction_classification.csv"),
    index=False
)

df_clim_classification


# Calculate the percentage of each case

# In[ ]:


# ============================================================
# Percentage of each correction behavior class
# ============================================================

df_class_percent = (
    df_clim_classification
    .value_counts("correction_class")
    .rename_axis("correction_class")
    .reset_index(name="n_cases")
)

df_class_percent["percentage"] = (
    df_class_percent["n_cases"]
    / df_class_percent["n_cases"].sum()
    * 100
)

df_class_percent.to_csv(
    os.path.join(TABLE_METRICS_PATH, "sst_climatology_correction_class_percentages.csv"),
    index=False
)

df_class_percent


# ### Previous code improved

# In[133]:


# ============================================================
# Climatological pattern diagnostics
# ============================================================

def climatology_pattern_metrics(original_clim, corrected_clim, obs_clim):
    """
    Compute diagnostics of mean error, seasonal shape and seasonal amplitude.
    """

    df = pd.concat(
        [
            original_clim.rename("original"),
            corrected_clim.rename("corrected"),
            obs_clim.rename("obs")
        ],
        axis=1,
        join="inner"
    ).dropna()

    if len(df) == 0:
        return {
            "n_months": 0,
            "mae_original_clim": np.nan,
            "mae_corrected_clim": np.nan,
            "relative_improvement_clim_percent": np.nan,
            "corr_original_shape": np.nan,
            "corr_corrected_shape": np.nan,
            "amp_obs": np.nan,
            "amp_original": np.nan,
            "amp_corrected": np.nan,
            "amp_ratio_original": np.nan,
            "amp_ratio_corrected": np.nan,
            "amp_error_original": np.nan,
            "amp_error_corrected": np.nan,
            "std_obs": np.nan,
            "std_original": np.nan,
            "std_corrected": np.nan,
            "std_ratio_original": np.nan,
            "std_ratio_corrected": np.nan
        }

    # Mean absolute errors
    mae_original = np.mean(np.abs(df["original"] - df["obs"]))
    mae_corrected = np.mean(np.abs(df["corrected"] - df["obs"]))

    if mae_original == 0:
        relative_improvement = np.nan
    else:
        relative_improvement = (
            (mae_original - mae_corrected) / mae_original * 100
        )

    # Remove annual mean to evaluate seasonal shape
    original_shape = df["original"] - df["original"].mean()
    corrected_shape = df["corrected"] - df["corrected"].mean()
    obs_shape = df["obs"] - df["obs"].mean()

    # Correlation of seasonal shape
    corr_original_shape = original_shape.corr(obs_shape)
    corr_corrected_shape = corrected_shape.corr(obs_shape)

    # Seasonal amplitude: max - min
    amp_obs = df["obs"].max() - df["obs"].min()
    amp_original = df["original"].max() - df["original"].min()
    amp_corrected = df["corrected"].max() - df["corrected"].min()

    amp_ratio_original = amp_original / amp_obs if amp_obs != 0 else np.nan
    amp_ratio_corrected = amp_corrected / amp_obs if amp_obs != 0 else np.nan

    amp_error_original = abs(amp_ratio_original - 1)
    amp_error_corrected = abs(amp_ratio_corrected - 1)

    # Standard deviation of monthly climatology as another shape/amplitude metric
    std_obs = df["obs"].std()
    std_original = df["original"].std()
    std_corrected = df["corrected"].std()

    std_ratio_original = std_original / std_obs if std_obs != 0 else np.nan
    std_ratio_corrected = std_corrected / std_obs if std_obs != 0 else np.nan

    return {
        "n_months": len(df),
        "mae_original_clim": mae_original,
        "mae_corrected_clim": mae_corrected,
        "relative_improvement_clim_percent": relative_improvement,
        "corr_original_shape": corr_original_shape,
        "corr_corrected_shape": corr_corrected_shape,
        "amp_obs": amp_obs,
        "amp_original": amp_original,
        "amp_corrected": amp_corrected,
        "amp_ratio_original": amp_ratio_original,
        "amp_ratio_corrected": amp_ratio_corrected,
        "amp_error_original": amp_error_original,
        "amp_error_corrected": amp_error_corrected,
        "std_obs": std_obs,
        "std_original": std_original,
        "std_corrected": std_corrected,
        "std_ratio_original": std_ratio_original,
        "std_ratio_corrected": std_ratio_corrected
    }


# Stricter classification function

# In[134]:


# ============================================================
# Refined classification using mean error, shape and amplitude
# ============================================================

def classify_climatology_pattern(
    mae_original,
    mae_corrected,
    relative_improvement,
    corr_original_shape,
    corr_corrected_shape,
    amp_ratio_original,
    amp_ratio_corrected,
    close_mae_threshold=0.30,
    success_mae_threshold=0.30,
    min_relative_improvement_success=30.0,
    min_corr_success=0.85,
    amp_ratio_min=0.70,
    amp_ratio_max=1.30,
    degraded_amp_min=0.50
):
    """
    Classify climatological correction behavior.

    This classification is stricter than MAE-only classification because it
    also checks whether the corrected climatology preserves the seasonal cycle.
    """

    values = [
        mae_original,
        mae_corrected,
        relative_improvement,
        corr_corrected_shape,
        amp_ratio_corrected
    ]

    if any(not np.isfinite(v) for v in values):
        return "Invalid or missing data"

    # Case 1: original already close to observations
    if (
        mae_original <= close_mae_threshold
        and np.isfinite(corr_original_shape)
        and corr_original_shape >= min_corr_success
        and amp_ratio_min <= amp_ratio_original <= amp_ratio_max
    ):
        return "Original already close"

    # Case 2: corrected series is close, shape is good, amplitude is realistic
    if (
        mae_corrected <= success_mae_threshold
        and relative_improvement >= min_relative_improvement_success
        and corr_corrected_shape >= min_corr_success
        and amp_ratio_min <= amp_ratio_corrected <= amp_ratio_max
    ):
        return "Successful correction"

    # Case 3: mean improved, but seasonal amplitude was flattened or distorted
    if (
        relative_improvement >= min_relative_improvement_success
        and (
            corr_corrected_shape < min_corr_success
            or amp_ratio_corrected < amp_ratio_min
            or amp_ratio_corrected > amp_ratio_max
        )
    ):
        return "Mean corrected but seasonal cycle degraded"

    # Case 4: some improvement, but not enough to call it successful
    if relative_improvement > 0:
        return "Partial correction"

    # Case 5: no improvement or degradation
    return "Correction failed"


# In[ ]:


# ============================================================
# Build refined climatological classification table
# ============================================================

area_name_map = area_display_name_map.copy()

classification_rows = []

for model in models:
    for area in areas:

        original = slice_series(
            series_historical[model][area],
            test_start,
            test_end
        )

        corrected = slice_series(
            hist_best[model][area],
            test_start,
            test_end
        )

        obs = slice_series(
            sst_obs[area],
            test_start,
            test_end
        )

        clim_original = monthly_climatology(original, test_start, test_end)
        clim_corrected = monthly_climatology(corrected, test_start, test_end)
        clim_obs = monthly_climatology(obs, test_start, test_end)

        metrics = climatology_pattern_metrics(
            original_clim=clim_original,
            corrected_clim=clim_corrected,
            obs_clim=clim_obs
        )

        correction_class = classify_climatology_pattern(
            mae_original=metrics["mae_original_clim"],
            mae_corrected=metrics["mae_corrected_clim"],
            relative_improvement=metrics["relative_improvement_clim_percent"],
            corr_original_shape=metrics["corr_original_shape"],
            corr_corrected_shape=metrics["corr_corrected_shape"],
            amp_ratio_original=metrics["amp_ratio_original"],
            amp_ratio_corrected=metrics["amp_ratio_corrected"],
            close_mae_threshold=0.30,
            success_mae_threshold=0.30,
            min_relative_improvement_success=30.0,
            min_corr_success=0.85,
            amp_ratio_min=0.70,
            amp_ratio_max=1.30,
            degraded_amp_min=0.50
        )

        row = {
            "model": model,
            "area": area,
            "area_name": area_name_map[area],
            "selected_method": best_method_used[model][area],
            "correction_class": correction_class
        }

        row.update(metrics)
        classification_rows.append(row)

df_clim_pattern_classification = pd.DataFrame(classification_rows)

df_clim_pattern_classification.to_csv(
    os.path.join(TABLE_METRICS_PATH, "sst_climatology_pattern_classification.csv"),
    index=False
)

df_clim_pattern_classification


# In[136]:


df_clim_pattern_classification[18:35]


# Percentage of classes

# In[ ]:


# ============================================================
# Percentage of each refined class
# ============================================================

df_class_percent_refined = (
    df_clim_pattern_classification
    .value_counts("correction_class")
    .rename_axis("correction_class")
    .reset_index(name="n_cases")
)

df_class_percent_refined["percentage"] = (
    df_class_percent_refined["n_cases"]
    / df_class_percent_refined["n_cases"].sum()
    * 100
)

df_class_percent_refined.to_csv(
    os.path.join(TABLE_METRICS_PATH, "sst_climatology_pattern_class_percentages.csv"),
    index=False
)

df_class_percent_refined


# Performance matrix

# In[ ]:


# ============================================================
# Heatmap: refined correction behavior class by model and area
# ============================================================

class_order = {
    "Original already close": 0,
    "Successful correction": 1,
    "Mean corrected but seasonal cycle degraded": 2,
    "Partial correction": 3,
    "Correction failed": 4,
    "Invalid or missing data": 5
}

class_label_map = {
    0: "Original close",
    1: "Successful",
    2: "Mean only",
    3: "Partial",
    4: "Failed",
    5: "Invalid"
}

df_clim_pattern_classification["class_code"] = (
    df_clim_pattern_classification["correction_class"].map(class_order)
)

heatmap_class = (
    df_clim_pattern_classification
    .pivot(index="model", columns="area", values="class_code")
    .reindex(index=models, columns=areas)
)

fig, ax = plt.subplots(figsize=(15, 4.8))

cmap = plt.cm.get_cmap("Set2", len(class_order))

im = ax.imshow(
    heatmap_class.values,
    aspect="auto",
    cmap=cmap,
    vmin=-0.5,
    vmax=len(class_order) - 0.5
)

ax.set_xticks(np.arange(len(areas)))
ax.set_yticks(np.arange(len(models)))

ax.set_xticklabels([area_name_map[a] for a in areas], rotation=45, ha="right")
ax.set_yticklabels(models)

# for i in range(len(models)):
#     for j in range(len(areas)):
#         value = heatmap_class.iloc[i, j]
#         if pd.notna(value):
#             ax.text(
#                 j,
#                 i,
#                 class_label_map[int(value)],
#                 ha="center",
#                 va="center",
#                 fontsize=7
#             )

cbar = plt.colorbar(im, ax=ax, ticks=list(class_label_map.keys()))
cbar.ax.set_yticklabels([class_label_map[i] for i in class_label_map.keys()])
cbar.set_label("Correction behavior class")

ax.set_title("Refined classification of climatological correction behavior")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_SUMMARY_PATH, "heatmap_refined_climatology_correction_behavior_class.png"),
    format="png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

