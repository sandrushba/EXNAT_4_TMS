import os
#os.environ["JAX_PLATFORMS"] = "cpu"
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import hssm
import pandas as pd
import arviz as az
import pytensor
import matplotlib.pyplot as plt
import numpy as np
#import multiprocessing as mp

##########################################################
# Set-up simple coding scheme for factors
def add_simple_contrasts(df, var, levels, prefix=None, drop_original=False):
    df = df.copy()
    prefix = var if prefix is None else prefix

    x = pd.Categorical(df[var], categories=levels, ordered=True)
    nlevels = len(levels)

    if x.isna().any():
        bad = df.loc[pd.isna(x), var].unique()
        raise ValueError(f"{var}: values not found in levels list: {bad}")

    for j, level in enumerate(levels[1:], start=1):
        col = f"{prefix}_simp_{level}"
        vals = np.full(len(df), -1.0 / nlevels, dtype=float)
        vals[np.asarray(x == level)] = (nlevels - 1.0) / nlevels
        df[col] = vals

    if drop_original:
        df = df.drop(columns=[var])

    return df
##########################################################

base_dir = "/data/p_03049/EXNAT_4_TMS/HSSM/"

# load data
data_file = os.path.join(base_dir, "Data", "HSSM_dual_task_df.csv")
df_dual_task = pd.read_csv(data_file) # continuous predictors are alread mean-centered

##########################################################
## Set reference levels for categorical data
df_dual_task["stim_type"] = pd.Categorical(
    df_dual_task["stim_type"],
    categories=["Sham", "Offline", "Offline + Online"],
)

df_dual_task["session"] = pd.Categorical(
    df_dual_task["session"].astype(str),
    categories=["1", "2", "3"],
)

df_dual_task["is_trigger"] = pd.Categorical(
    df_dual_task["is_trigger"].astype(str),
    categories=["False", "True"],
)

df_model = df_dual_task.copy()

df_model = add_simple_contrasts(
    df_model, "stim_type",
    levels=["Sham", "Offline", "Offline + Online"]
)

df_model = add_simple_contrasts(
    df_model, "is_trigger",
    levels=["False", "True"]
)

df_model = add_simple_contrasts(
    df_model, "session",
    levels=["1", "2", "3"]
)

##########################################################
## * BASELINE MODEL
model_dual_task_baseline = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
)

# graphical illustration of model
#graph = model_nback_baseline.graph(name="BL_nback")
#graph.view()

ddm_dual_task_baseline = model_dual_task_baseline.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

az.to_netcdf(model_dual_task_baseline._inference_obj, f'HSSM/hssm_models/ddm_dual_task_baseline.nc')
model_dual_task_baseline.save_model(model_name="dual_task_baseline")

model_summary_BL = model_dual_task_baseline.summary()
#### trace plots
model_dual_task_baseline.plot_trace(show=True)
hssm.plotting.plot_predictive(model_dual_task_baseline)
hssm.plotting.plot_model_cartoon(model_dual_task_baseline)
plt.show()

##########################################################
## * V FULL MODEL REDUCED
model_dual_task_v_full_red = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "identity",
        },
    ],
)

# sample model
ddm_dual_task_v_full_red = model_dual_task_v_full_red.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_dual_task_v_full_red.save_model(model_name="dual_task_v_full_red")

model_summary = model_dual_task_v_full_red.summary()
model_dual_task_v_full_red = az.summary(model_dual_task_v_full_red.traces,
                                        var_names=['v_C(stim_type)',
                                                   'v_C(task)',
                                                   'v_C(session)',
                                                   'v_C(is_trigger)',
                                                   'v_C(stim_type):C(task)',
                                                   't','z','a'])

##########################################################
## * V FULL MODEL FULL – FULL RANDOM SLOPE
model_dual_task_v_full_slopes = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + (1|participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + (1|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_dual_task_v_full_slopes = model_dual_task_v_full_slopes.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_dual_task_v_full_slopes.save_model(model_name="dual_task_v_full_slopes")

model_summary = model_dual_task_v_full_slopes.summary()
model_dual_task_v_full_red = az.summary(model_dual_task_v_full_red.traces,
                                        var_names=['v_C(stim_type)',
                                                   'v_C(task)',
                                                   'v_C(session)',
                                                   'v_C(is_trigger)',
                                                   'v_C(stim_type):C(task)',
                                                   't','z','a'])

##########################################################
## * VAT MODEL (a + t only random intercepts)
model_dual_task_vat = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1|participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_dual_task_vat = model_dual_task_vat.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_dual_task_vat.save_model(model_name="dual_task_vat")

model_summary = model_dual_task_v_full_slopes.summary()
model_dual_task_v_full_red = az.summary(model_dual_task_v_full_red.traces,
                                        var_names=['v_C(stim_type)',
                                                   'v_C(task)',
                                                   'v_C(session)',
                                                   'v_C(is_trigger)',
                                                   'v_C(stim_type):C(task)',
                                                   't','z','a'])

##########################################################
## * VA MODEL (a only random intercept)
model_dual_task_va = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1|participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + "
                       "(1|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_dual_task_va = model_dual_task_va.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_dual_task_va.save_model(model_name="dual_task_va")

model_summary = model_dual_task_v_full_slopes.summary()
model_dual_task_v_full_red = az.summary(model_dual_task_v_full_red.traces,
                                        var_names=['v_C(stim_type)',
                                                   'v_C(task)',
                                                   'v_C(session)',
                                                   'v_C(is_trigger)',
                                                   'v_C(stim_type):C(task)',
                                                   't','z','a'])

##########################################################
## * VA MODEL SLOPES (a with slope)
model_dual_task_va_slopes = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + "
                       "(1|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_dual_task_va_slopes = model_dual_task_va_slopes.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_dual_task_va_slopes.save_model(model_name="dual_task_va_slopes")

model_summary = model_dual_task_v_full_slopes.summary()
model_dual_task_v_full_red = az.summary(model_dual_task_v_full_red.traces,
                                        var_names=['v_C(stim_type)',
                                                   'v_C(task)',
                                                   'v_C(session)',
                                                   'v_C(is_trigger)',
                                                   'v_C(stim_type):C(task)',
                                                   't','z','a'])

##########################################################
## * VT MODEL (t only random intercept)
model_dual_task_vt = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + "
                       "(1|participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_dual_task_vt = model_dual_task_vt.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_dual_task_vt.save_model(model_name="dual_task_vt")

model_summary = model_dual_task_v_full_slopes.summary()
model_dual_task_v_full_red = az.summary(model_dual_task_v_full_red.traces,
                                        var_names=['v_C(stim_type)',
                                                   'v_C(task)',
                                                   'v_C(session)',
                                                   'v_C(is_trigger)',
                                                   'v_C(stim_type):C(task)',
                                                   't','z','a'])

##########################################################
## * VT MODEL SLOPES (t with slope for stim cond)
model_dual_task_vt_slopes = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + "
                       "(1|participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_dual_task_vt_slopes = model_dual_task_vt_slopes.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_dual_task_vt_slopes.save_model(model_name="dual_task_vt_slopes")

model_summary = model_dual_task_v_full_slopes.summary()
model_dual_task_v_full_red = az.summary(model_dual_task_v_full_red.traces,
                                        var_names=['v_C(stim_type)',
                                                   'v_C(task)',
                                                   'v_C(session)',
                                                   'v_C(is_trigger)',
                                                   'v_C(stim_type):C(task)',
                                                   't','z','a'])

##########################################################
## * VZ MODEL SLOPES (z with slope)
model_dual_task_vz_slopes = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + "
                       "stim_type_simp_Offline * surprisal_2 + "
                       "'stim_type_simp_Offline + Online' * surprisal_2 + "
                       "entropy_2 + zipf_frequency + word_length_single + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + stim_type_simp_Offline + 'stim_type_simp_Offline + Online'|participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + "
                       "(1|participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + "
                       "(1|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_dual_task_vz_slopes = model_dual_task_vz_slopes.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_dual_task_vz_slopes.save_model(model_name="dual_task_vz_slopes")

model_summary = model_dual_task_v_full_slopes.summary()
model_dual_task_v_full_red = az.summary(model_dual_task_v_full_red.traces,
                                        var_names=['v_C(stim_type)',
                                                   'v_C(task)',
                                                   'v_C(session)',
                                                   'v_C(is_trigger)',
                                                   'v_C(stim_type):C(task)',
                                                   't','z','a'])
