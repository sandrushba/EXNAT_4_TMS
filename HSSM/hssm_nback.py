import os
#os.environ["JAX_PLATFORMS"] = "cpu"
#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import hssm
import pandas as pd
import arviz as az
import pytensor
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp

from hssm.plotting import plot_model_cartoon


#mp.set_start_method("spawn", force=True)


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
data_file = os.path.join(base_dir, "Data", "HSSM_nback_df.csv")
df_single_nback = pd.read_csv(data_file)

# In HSSM’s default DDM specification, v receives a Normal(0, 2) prior, a a HalfNormal(2) prior, z a Uniform(0,
# 1) prior, and t a HalfNormal(2) prior.

##########################################################
## Set reference levels for categorical data
df_single_nback["task"] = pd.Categorical(
    df_single_nback["task"],
    categories=["1back_single_main_click", "2back_single_main_click"],
)

df_single_nback["stim_type"] = pd.Categorical(
    df_single_nback["stim_type"],
    categories=["Sham", "Offline", "Offline + Online"],
)

df_single_nback["session"] = pd.Categorical(
    df_single_nback["session"].astype(str),
    categories=["1", "2", "3"],
)

df_single_nback["is_trigger"] = pd.Categorical(
    df_single_nback["is_trigger"].astype(str),
    categories=["False", "True"],
)

df_model = df_single_nback.copy()

df_model = add_simple_contrasts(
    df_model, "task",
    levels=["1back_single_main_click", "2back_single_main_click"]
)

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

# mean-center time
df_model["time_from_onset_task_c"] = df_model["time_from_onset_task"] - df_model["time_from_onset_task"].mean()

##########################################################
## * BASELINE MODEL
model_nback_baseline = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
)

# graphical illustration of model
graph = model_nback_baseline.graph(name="BL_nback")
graph.view()

ddm_nback_baseline = model_nback_baseline.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    #cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

model_summary_BL = model_nback_baseline.summary()
#### trace plots
model_nback_baseline.plot_trace(show=True)
hssm.plotting.plot_predictive(model_nback_baseline)
hssm.plotting.plot_model_cartoon(model_nback_baseline)
plt.show()

model_nback_baseline.save_model(model_name="nback_baseline")

##########################################################
## * V FULL MODEL REDUCED
model_nback_v_full_red = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
            "link": "identity",
        },
    ],
)

# sample model
ddm_nback_v_full_red = model_nback_v_full_red.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 500, # samples to keep
    tune = 500, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=2,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    #chain_method="parallel",
)

model_summary = model_nback_v_full_red.summary()

model_nback_v_full_red_summary = az.summary(model_nback_v_full_red.traces, var_names=['v_C(stim_type)','v_C(task)','v_C(session)','v_C(is_trigger)',
                                                     'v_C(stim_type):C(task)','t','z','a'])
# save model
model_nback_v_full_red.save_model(model_name="nback_v_full_red")

##########################################################
## * V MODEL FULL – FULL RANDOM SLOPE
model_nback_v_full_slopes = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
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
ddm_nback_v_full_slopes = model_nback_v_full_slopes.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

full_summary=model_nback_v_full_slopes.summary()
#model_summary = model_nback_v.summary()
model_nback_v_full_slopes_summary = az.summary(ddm_nback_v_full_slopes,
                                           var_names=['v_stim_type_simp_Offline',
                                                      'v_stim_type_simp_Offline + Online',
                                                      'v_task_simp_2back_single_main_click',
                                                      'v_stim_type_simp_Offline + Online:task_simp_2back_single_main_click',
                                                      'v_stim_type_simp_Offline:task_simp_2back_single_main_click',
                                                      'v_session_simp_2',
                                                      'v_session_simp_3',
                                                      'v_is_trigger_simp_True',
                                                      'v_Intercept',
                                                      't_Intercept',
                                                      'z_Intercept',
                                                      'a_Intercept'])

# save model
model_nback_v_full_slopes.save_model(model_name="nback_v_full_slopes")

##########################################################
## * VAT MODEL FULL – FULL RANDOM SLOPE V
model_nback_vat_full_slopes = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + (1|participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + (1|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_nback_vat_full_slopes = model_nback_vat_full_slopes.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 3000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    #chain_method="parallel",
)

# save model
model_nback_vat_full_slopes.save_model(model_name="nback_vat_full_slopes")

full_summary=model_nback_v_full_slopes.summary()
#model_summary = model_nback_v.summary()
model_nback_v_full_slopes_summary = az.summary(ddm_nback_v_full_slopes,
                                           var_names=['v_stim_type_simp_Offline',
                                                      'v_stim_type_simp_Offline + Online',
                                                      'v_task_simp_2back_single_main_click',
                                                      'v_stim_type_simp_Offline + Online:task_simp_2back_single_main_click',
                                                      'v_stim_type_simp_Offline:task_simp_2back_single_main_click',
                                                      'v_session_simp_2',
                                                      'v_session_simp_3',
                                                      'v_is_trigger_simp_True',
                                                      'v_Intercept',
                                                      't_Intercept',
                                                      'z_Intercept',
                                                      'a_Intercept'])

##########################################################
## * VA MODEL
model_nback_va = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
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
                       "stim_type_simp_Offline + 'stim_type_simp_Offline + Online' + "
                       "task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
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
ddm_nback_va = model_nback_va.sample(
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
model_nback_va.save_model(model_name="nback_va")

# model summary
model_summary = model_nback_va.summary()
model_nback_va_summary = az.summary(ddm_nback_va,
                                           var_names=['v_stim_type_simp_Offline',
                                                      'v_stim_type_simp_Offline + Online',
                                                      'v_task_simp_2back_single_main_click',
                                                      'v_stim_type_simp_Offline + Online:task_simp_2back_single_main_click',
                                                      'v_stim_type_simp_Offline:task_simp_2back_single_main_click',
                                                      'v_session_simp_2',
                                                      'v_session_simp_3',
                                                      'v_is_trigger_simp_True',
                                                      'v_time_from_onset_task_c',
                                                      'v_Intercept',
                                                      'a_stim_type_simp_Offline',
                                                      'a_stim_type_simp_Offline + Online',
                                                      'a_task_simp_2back_single_main_click',
                                                      #'a_stim_type_simp_Offline + Online:task_simp_2back_single_main_click',
                                                      #'a_stim_type_simp_Offline:task_simp_2back_single_main_click',
                                                      'a_session_simp_2',
                                                      'a_session_simp_3',
                                                      'a_is_trigger_simp_True',
                                                      'a_time_from_onset_task_c',
                                                      'a_Intercept',
                                                      't_Intercept',
                                                      'z_Intercept',])

# trace plots
model_nback_va.plot_trace(show=True)

# plot posterior predictives
hssm.plotting.plot_predictive(model_nback_va)
plt.show()

hssm.plotting.plot_model_cartoon(model_nback_va)
plt.show()

##########################################################
## * VA MODEL WITH INTERACTION OF FIXED EFFECTS IN A
model_nback_va_int = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
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
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
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
ddm_nback_va_int = model_nback_va_int.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    #cores=4,
    #chains=4,
    #sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    #chain_method="vectorized",
)

# save model
model_nback_va_int.save_model(model_name="nback_va_int")

# model summary
model_summary = model_nback_va_slope_int.summary()
model_nback_va_summary = az.summary(ddm_nback_va,
                                           var_names=['v_stim_type_simp_Offline',
                                                      'v_stim_type_simp_Offline + Online',
                                                      'v_task_simp_2back_single_main_click',
                                                      'v_stim_type_simp_Offline + Online:task_simp_2back_single_main_click',
                                                      'v_stim_type_simp_Offline:task_simp_2back_single_main_click',
                                                      'v_session_simp_2',
                                                      'v_session_simp_3',
                                                      'v_is_trigger_simp_True',
                                                      'v_time_from_onset_task_c',
                                                      'v_Intercept',
                                                      'a_stim_type_simp_Offline',
                                                      'a_stim_type_simp_Offline + Online',
                                                      'a_task_simp_2back_single_main_click',
                                                      #'a_stim_type_simp_Offline + Online:task_simp_2back_single_main_click',
                                                      #'a_stim_type_simp_Offline:task_simp_2back_single_main_click',
                                                      'a_session_simp_2',
                                                      'a_session_simp_3',
                                                      'a_is_trigger_simp_True',
                                                      'a_time_from_onset_task_c',
                                                      'a_Intercept',
                                                      't_Intercept',
                                                      'z_Intercept',])

# trace plots
model_nback_va.plot_trace(show=True)

# plot posterior predictives
hssm.plotting.plot_predictive(model_nback_va)
plt.show()

hssm.plotting.plot_model_cartoon(model_nback_va)
plt.show()

##########################################################
## * VA_TRIGGER MODEL WITH INTERACTION OF STIMTYPE (OFFLINE + ONLINE) AND TRIGGER
model_nback_va_trigger = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True * 'stim_type_simp_Offline + Online' + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
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
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True * 'stim_type_simp_Offline + Online' + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
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
ddm_nback_va_trigger = model_nback_va_trigger.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws=1000, # samples to keep
    tune=2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=4,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_nback_va_trigger.save_model(model_name="nback_va_trigger")

# model summary
model_summary = model_nback_va_trigger.summary()

# trace plots
model_nback_va_trigger.plot_trace(show=True)

# plot posterior predictives
hssm.plotting.plot_predictive(model_nback_va_trigger)
plt.show()

hssm.plotting.plot_model_cartoon(model_nback_va_trigger, n_trajectories=10, plot_predictive_mean=True)
plt.show()
plt.savefig("nback_va_trigger.pdf", dpi=300)

hssm.plotting.plot_model_cartoon(model_nback_va_trigger,
                                 bins=30,
                                 n_trajectories=10,
                                 plot_predictive_mean=True)

##########################################################
## * VT MODEL
model_nback_vt = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
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
            "formula": "t ~ 1 + "
                "stim_type_simp_Offline + 'stim_type_simp_Offline + Online' + "
                "task_simp_2back_single_main_click + "
                "is_trigger_simp_True + "
                "session_simp_2 + session_simp_3 +"
                "time_from_onset_task_c +"
                "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
                "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_nback_vt = model_nback_vt.sample(
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
model_nback_vt.save_model(model_name="nback_vt_int")

##########################################################
## * VT MODEL WITH INTERACTION OF FIXED EFFECTS IN T
model_nback_vt_int = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant)",
            "link": "logit",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
            "link": "log",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + (1|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_nback_vt_int = model_nback_vt_int.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=4,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_nback_vt_int.save_model(model_name="nback_vt_int")

# model summary
model_summary = model_nback_va_slope_int.summary()
model_nback_va_summary = az.summary(ddm_nback_va,
                                           var_names=['v_stim_type_simp_Offline',
                                                      'v_stim_type_simp_Offline + Online',
                                                      'v_task_simp_2back_single_main_click',
                                                      'v_stim_type_simp_Offline + Online:task_simp_2back_single_main_click',
                                                      'v_stim_type_simp_Offline:task_simp_2back_single_main_click',
                                                      'v_session_simp_2',
                                                      'v_session_simp_3',
                                                      'v_is_trigger_simp_True',
                                                      'v_time_from_onset_task_c',
                                                      'v_Intercept',
                                                      'a_stim_type_simp_Offline',
                                                      'a_stim_type_simp_Offline + Online',
                                                      'a_task_simp_2back_single_main_click',
                                                      #'a_stim_type_simp_Offline + Online:task_simp_2back_single_main_click',
                                                      #'a_stim_type_simp_Offline:task_simp_2back_single_main_click',
                                                      'a_session_simp_2',
                                                      'a_session_simp_3',
                                                      'a_is_trigger_simp_True',
                                                      'a_time_from_onset_task_c',
                                                      'a_Intercept',
                                                      't_Intercept',
                                                      'z_Intercept',])

# trace plots
model_nback_va.plot_trace(show=True)

# plot posterior predictives
hssm.plotting.plot_predictive(model_nback_va)
plt.show()

hssm.plotting.plot_model_cartoon(model_nback_va)
plt.show()

##########################################################
## * VZ MODEL WITH INTERACTION OF FIXED EFFECTS IN Z
model_nback_vz_int = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
            "link": "logit",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + (1|participant)",
            "link": "log",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + (1|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_nback_vz_int = model_nback_vz_int.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=4,
    chains=4,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_nback_vz_int.save_model(model_name="nback_vz_int")

# model summary
model_summary = model_nback_va_slope_int.summary()

# trace plots
model_nback_vz_int.plot_trace(show=True)

# plot posterior predictives
hssm.plotting.plot_predictive(model_nback_vz_int)
plt.show()

hssm.plotting.plot_model_cartoon(model_nback_vz_int, n_trajectories = 10)
plt.show()

##########################################################
## * VAT MODEL (a + t only random intercepts)
model_nback_vat = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "(1|participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "(1|participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_nback_vat= model_nback_vat.sample(
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
model_nback_vat.save_model(model_name="nback_vat")

##########################################################
## * VAT MODEL FULL – FULL RANDOM SLOPE VAT
## Doesn't sample well, way too overparametrized
model_nback_vat_full_slopes_all = hssm.HSSM(
    data=df_model,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "is_trigger_simp_True + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task_c +"
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + (1|participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + stim_type_simp_Offline * task_simp_2back_single_main_click + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "(1 + task_simp_2back_single_main_click + stim_type_simp_Offline + 'stim_type_simp_Offline + Online' |participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_nback_vat_full_slopes_all= model_nback_vat_full_slopes_all.sample(
    idata_kwargs=dict(log_likelihood=True),
    target_accept=0.9,
    draws = 1000, # samples to keep
    tune = 2000, # samples for adaptation, discarded by default, don't use for inference (equivalent to burn)
    cores=1,
    chains=2,
    sampler="numpyro", #use JAX-compiled NUTS for speed (vs. default PyMC sampler)
    chain_method="vectorized",
)

# save model
model_nback_vat_full_slopes.save_model(model_name="nback_vat_full_slopes")

##########################################################
# Prepare df for online trials only
df_single_nback_online = df_single_nback[df_single_nback['is_trigger'] == 'True']

## Set reference levels for categorical data
df_single_nback_online["stim_type"] = pd.Categorical(
    df_single_nback_online["stim_type"],
    categories=["Sham", "Offline + Online"],
)

df_model_online = df_single_nback_online.copy()

df_model_online = add_simple_contrasts(
    df_model_online, "task",
    levels=["1back_single_main_click", "2back_single_main_click"]
)

df_model_online = add_simple_contrasts(
    df_model_online, "stim_type",
    levels=["Sham", "Offline + Online"]
)

df_model_online = add_simple_contrasts(
    df_model_online, "session",
    levels=["1", "2", "3"]
)

##########################################################
## * VA MODEL FULL – ONLY ONLINE TRIALS (ONLY FOR OFFLINE+ONLINE AND SHAM SESSIONS)
model_nback_va_online = hssm.HSSM(
    data=df_model_online,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
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
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
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
ddm_nback_va_online = model_nback_va_online.sample(
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
model_nback_va_online.save_model(model_name="nback_va_online")

# trace plots
model_nback_va_online.plot_trace(show=True)

# plot posterior predictives
hssm.plotting.plot_predictive(model_nback_va)
plt.show()

hssm.plotting.plot_model_cartoon(model_nback_va)
plt.show()

##########################################################
## * VAZ MODEL FULL – ONLY ONLINE TRIALS (ONLY FOR OFFLINE+ONLINE AND SHAM SESSIONS)
model_nback_vaz_online = hssm.HSSM(
    data=df_model_online,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
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
ddm_nback_vaz_online = model_nback_vaz_online.sample(
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
model_nback_vaz_online.save_model(model_name="nback_vaz_online")

##########################################################
## * VAT MODEL FULL – ONLY ONLINE TRIALS (ONLY FOR OFFLINE+ONLINE AND SHAM SESSIONS)
model_nback_vat_online = hssm.HSSM(
    data=df_model_online,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
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
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_nback_vat_online = model_nback_vat_online.sample(
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
model_nback_vat_online.save_model(model_name="nback_vat_online")

##########################################################
## * VATZ MODEL FULL – ONLY ONLINE TRIALS (ONLY FOR OFFLINE+ONLINE AND SHAM SESSIONS)
model_nback_vatz_online = hssm.HSSM(
    data=df_model_online,
    model="ddm",
    prior_settings="safe",
    noncentered=True,
    include=[
        {
            "name": "v",
            "formula": "v ~ 1 + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
            "link": "identity",
        },
        {
            "name": "z",
            "formula": "z ~ 1 + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
            "link": "logit",
        },
        {
            "name": "a",
            "formula": "a ~ 1 + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
            "link": "log",
        },
        {
            "name": "t",
            "formula": "t ~ 1 + "
                       "'stim_type_simp_Offline + Online' * task_simp_2back_single_main_click + "
                       "session_simp_2 + session_simp_3 +"
                       "time_from_onset_task +"
                       "(1 + task_simp_2back_single_main_click + 'stim_type_simp_Offline + Online' |participant)",
            "link": "log",
        },
    ],
    p_outlier=0.05,
)

# sample model
ddm_nback_vatz_online = model_nback_vatz_online.sample(
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
model_nback_vatz_online.save_model(model_name="nback_vatz_online")