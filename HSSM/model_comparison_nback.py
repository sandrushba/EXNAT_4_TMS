import os
#os.environ["JAX_PLATFORMS"] = "cpu" # when CUDA and GPUs aren't available
import hssm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
#import multiprocessing as mp
import seaborn as sns
import hssm.plotting
from graphviz import parameters
from pymc import sample_posterior_predictive

#from hssm_nback import df_model, df_single_nback

base_dir = "/data/p_03049/EXNAT_4_TMS/HSSM/hssm_models"

#  ---------------------------------------------------
# * LOAD MODELS
#  ---------------------------------------------------
def load_models(model_paths, base_dir=base_dir):
    models = {}

    for model_name, path in model_paths.items():
        try:
            full_path = os.path.join(base_dir, path) if base_dir else path
            models[model_name] = hssm.HSSM.load_model(path=full_path)
            print(f"Successfully loaded model: {model_name}")
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")

    return models

model_paths = {
    "nback_baseline": "nback_baseline",
    "nback_v_full_red": "nback_v_full_red",
    "nback_v_full_slopes": "nback_v_full_slopes",
    "nback_va": "nback_va",
    "nback_va_int": "nback_va_int",
    "nback_vt": "nback_vt",
    "nback_vt_int": "nback_vt_int",
    "nback_vz_int": "nback_vz_int",
    "nback_vat": "nback_vat",
    "nback_vat_full_slopes": "nback_vat_full_slopes",
    "nback_va_trigger": "nback_va_trigger",
}

models = load_models(model_paths)

#  ---------------------------------------------------
# MODEL COMPARISON
#  ---------------------------------------------------
model_comparison = az.compare(
    {
        "Model 1": models["nback_baseline"].traces, # model without any specification
        #"Model 2": models["nback_v_full_red"].traces, # model only with fixed and random effects for v
        "Model 3": models["nback_v_full_slopes"].traces, # model incl full slope structure for v, random intercepts for a, t, z
        #"Model 4": models["nback_va"].traces, # model with regressors for v + a, full random structure for both but no interaction in main effects for a
        "Model 5": models["nback_va_int"].traces, # model with regressors for v + a with int ***
        #"Model 6": models["nback_vt"].traces, # model with regressors for v + t
        "Model 7": models["nback_vt_int"].traces, # model with regressors for v + t with int
        #"Model 8": models["nback_vat"].traces  # model with regressors for v, a, t, and full slopes for v
        "Model 9": models["nback_vat_full_slopes"].traces, # model with regressors for v, a, t, and full slopes for v
        "Model 10": models["nback_vz_int"].traces, # model with regressors for v + z
        "Model 11": models["nback_va_trigger"].traces # model with regressors for v, a, t, and full slopes for v
    },
    ic="loo",
)
model_comparison.to_csv("/data/p_03049/EXNAT_4_TMS/HSSM/model_comparison_nback_loo.csv")

az.plot_compare(model_comparison, figsize=(12, 4))
plt.show()

# if you need to compute log-likelihood post-hoc for a model
import pymc as pm
idata = pm.compute_log_likelihood(
    models['nback_vat_full_slopes'].traces,
    model=models['nback_vat_full_slopes'].pymc_model,
    extend_inferencedata=True,
    progressbar=True,
)

# --------------------------------------------------
# Compute pointwise PSIS-LOO
# --------------------------------------------------
loo_res = az.loo(models["nback_v_full"].traces, pointwise=True, var_name="rt,response")

print("\nLOO result:")
print(loo_res)

pareto_k = loo_res.pareto_k.values
good_k_threshold = float(loo_res.good_k)

print("\nPareto-k threshold used by ArviZ:", good_k_threshold)
print("Max Pareto-k:", np.max(pareto_k))
print("N bad (k > 0.7):", np.sum(pareto_k > 0.7))
print("N very bad (k > 1.0):", np.sum(pareto_k > 1.0))

# Identify problematic trial rows
bad_idx = np.where(pareto_k > 0.7)[0]

df_bad = df_single_nback.iloc[bad_idx].copy()
df_bad["pareto_k"] = pareto_k[bad_idx]
df_bad = df_bad.sort_values("pareto_k", ascending=False)

print("\nTop problematic observations:")
print(df_bad.head(20))

# Save if useful
df_bad.to_csv("pareto_k_flagged_trials.csv", index=False)

#  ---------------------------------------------------
# MODEL DIAGNOSTICS
#  ---------------------------------------------------
plot_dir = "/data/p_03049/EXNAT_4_TMS/HSSM/Plots"
## trace plots
output_dir = os.path.join(plot_dir, "trace_plots")

az.rcParams["plot.max_subplots"] = 200 # for forcing to plot all parameters
for model_name, model in models.items():
    idata = model._inference_obj
    all_vars = list(idata.posterior.data_vars)

    group_level_params = [var for var in all_vars if "participant" not in var]
    if len(group_level_params) > 0:
        az.plot_trace(idata, var_names=group_level_params)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_group_params_traces.png")
        plt.close()

    stim_level_params = [var for var in all_vars if "stim" in var and "participant" not in var]
    if len(stim_level_params) > 0:
        az.plot_trace(idata, var_names=stim_level_params)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_stim_params_traces.png")
        plt.close()

    subject_level_params = [var for var in all_vars if "participant" in var]
    if len(subject_level_params) > 0:
        az.plot_trace(idata, var_names=subject_level_params)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model_name}_subject_params_traces.png")
        plt.close()

##
output_dir = "/data/p_03049/EXNAT_4_TMS/HSSM/Model_summaries"

for model_name, model in models.items():
    idata = model._inference_obj
    all_vars = list(idata.posterior.data_vars)
    group_level_params = [
        var for var in all_vars if 'subj_idx' not in var
    ]
    group_level_summary = az.summary(idata, var_names=group_level_params)
    group_level_summary = group_level_summary.drop(columns=["mcse_mean", "mcse_sd"])
    group_level_summary.to_html(f"{output_dir}/{model_name}_summary.html")

model_stats = az.summary

#  ---------------------------------------------------
# PLOTTING
#  ---------------------------------------------------

# generate graph of winning model
graph = models['nback_va'].graph(name="N-back_va")
graph.view()

# plot posterior predictions of best models
hssm.plotting.plot_predictive(models["nback_va_int"])
plt.savefig(os.path.join(plot_dir, "ppc_nback_va_int.png"), dpi=150)
plt.show()

# plot posterior predictions of best models for each participant
hssm.plotting.plot_predictive(models["nback_va_int"], col="participant", col_wrap=5)
plt.savefig(os.path.join(plot_dir, "ppc_nback_va_int_participants.png"), dpi=150)
plt.show()
###############
# Specify a plot with one row and two columns
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot the posterior samples for condition 1
az.plot_posterior(
    models["nback_va_int"].traces.posterior, var_names=('v_stim_type_simp_Offline:task_simp_2back_single_main_click'),
    ax=ax, color="blue", hdi_prob=0.95
)

# Plot the posterior samples for condition 2
az.plot_posterior(models["nback_va_int"].traces.posterior, var_names=('v_stim_type_simp_Offline + Online:task_simp_2back_single_main_click'),
                  ax=ax, color="red", hdi_prob=0.95)

# Set x-axis limit
ax.set_xlim(-0.35, 0.35)

# Create proxy artists for the legend
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color="blue", label="2-back Offline"),
    Line2D([0], [0], color="red", label="2-back Offline + Online"),
]

# Add a legend
ax.legend(handles=legend_elements)

# Add a x-axis label
ax.set_xlabel("v")
# Percent of posterior samples
print(
    f"p(v_1 > v_2) = {np.mean(models['nback_va'].traces.posterior['v_stim_type_simp_Offline + Online:task_simp_2back_single_main_click'].values > models['nback_va'].traces.posterior['v_stim_type_simp_Offline:task_simp_2back_single_main_click'].values)}"
)
##############

# plot individual subjects:
hssm.plotting.plot_predictive(models["nback_v_full_slopes"])
plt.savefig(os.path.join(plot_dir, "ppc_nback_full_slopes.png"), dpi=150)

hssm.plotting.plot_predictive(models["nback_va_int"], col="participant", col_wrap=5)
plt.savefig(os.path.join(plot_dir, "ppc_nback_full_slopes_participants.png"), dpi=150)

az.plot_posterior(models["nback_va"].traces, var_names=('v_stim_type_simp_Offline:task_simp_2back_single_main_click'))
plt.show()

az.plot_posterior(models["nback_va"].traces, var_names=('a_task_simp_2back_single_main_click'))
plt.show()

az.plot_forest(models["nback_va"].traces, var_names=('a_stim_type_simp_Offline:task_simp_2back_single_main_click'),
               combined=False) # show i

#-------
posterior = models["nback_va_int"].traces.posterior

all_params = list(posterior.data_vars)
az.plot_posterior(posterior, var_names=all_params)
plt.show()


group_vars = [var for var in posterior.data_vars if "participant" not in var]
df = az.extract(models["nback_va_int"].traces, group="posterior", var_names=group_vars).to_dataframe()

output_dir = "/data/p_03049/EXNAT_4_TMS/HSSM/Model_summaries"
df.to_csv(os.path.join(output_dir, "nback_va_int.csv"))

#-------- extract subject-specific params ----
posterior = models["nback_va_int"].traces.posterior

all_params = list(posterior.data_vars)
subject_vars = [
    var for var in posterior.data_vars
    if "participant" in var
       and "sigma" not in var
       and "mu" not in var
       and "offset" not in var
]
summary = az.summary(
    posterior,
    var_names=subject_vars,
    stat_funcs=None,          # only default stats
    hdi_prob=0.95,            # or 0.89, etc.
    kind="stats",
)

# reformat summary a bit to a long format:
df = summary.reset_index().rename(columns={"index": "param_raw"})

def split_param(name):
    if "[" in name and name.endswith("]"):
        base, idx = name.split("[", 1)
        idx = idx[:-1]  # drop closing ']'
        return base, int(idx)
    else:
        return name, None

df[["parameter", "participant"]] = df["param_raw"].apply(
    lambda s: pd.Series(split_param(s))
)

# Keep only what you care about: mean + HDI and identifiers
cols_keep = ["parameter", "participant", "mean", "hdi_2.5%", "hdi_97.5%"]
df_subject_summary = df[cols_keep]

output_dir = "/data/p_03049/EXNAT_4_TMS/HSSM/Model_summaries"
df.to_csv(os.path.join(output_dir, "nback_va_int_subject_params.csv"))

#----------- extract subject parameters
idata = models["nback_va_int"].traces
print(list(idata.posterior.data_vars))

# Flatten chains × draws into a single samples dimension
post = az.extract(idata)  # xarray Dataset, dim "sample" = chains*draws

tms_vars = [v for v in idata.posterior.data_vars
            if "participant" in v and "stim_type" in v]
print(tms_vars)

# for offline TMS ----
# Fixed (population) TMS effect
v_tms_fixed = post["v_stim_type_simp_Offline"].values        # shape: (n_samples,)

# Random slope z-scores (noncentered)
v_tms_re_z  = post["v_stim_type_simp_Offline|participant"].values   # (n_subj, n_samples)
v_tms_re_sd = post["v_stim_type_simp_Offline|participant_sigma"].values  # (n_samples,)

# Reconstruct actual per-subject TMS slope
v_tms_re    = v_tms_re_z * v_tms_re_sd[np.newaxis, :]        # (n_subj, n_samples)

# Total subject-specific TMS effect on v:
# = group mean TMS effect + individual deviation
v_tms_subj  = v_tms_fixed[np.newaxis, :] + v_tms_re          # (n_subj, n_samples)

v_tms_mean  = v_tms_subj.mean(axis=1)   # point estimate per subject
v_tms_sd    = v_tms_subj.std(axis=1)    # posterior uncertainty per subject

a_tms_fixed = post["a_stim_type_simp_Offline"].values
a_tms_re_z  = post["a_stim_type_simp_Offline|participant"].values
a_tms_re_sd = post["a_stim_type_simp_Offline|participant_sigma"].values
a_tms_re    = a_tms_re_z * a_tms_re_sd[np.newaxis, :]
a_tms_subj_log = a_tms_fixed[np.newaxis, :] + a_tms_re  # log-scale slope
# Option A: keep on log scale (additive effect on log(a)) → correlate directly
a_tms_mean_log = a_tms_subj_log.mean(axis=1)

# for offline + online TMS
# Fixed (population) TMS effect
v_tms_fixed = post["v_stim_type_simp_Offline + Online"].values        # shape: (n_samples,)

# Random slope z-scores (noncentered)
v_tms_re_z  = post["v_stim_type_simp_Offline + Online|participant"].values   # (n_subj, n_samples)
v_tms_re_sd = post["v_stim_type_simp_Offline + Online|participant_sigma"].values  # (n_samples,)

# Reconstruct actual per-subject TMS slope
v_tms_re    = v_tms_re_z * v_tms_re_sd[np.newaxis, :]        # (n_subj, n_samples)

# Total subject-specific TMS effect on v:
# = group mean TMS effect + individual deviation
v_tms_subj  = v_tms_fixed[np.newaxis, :] + v_tms_re          # (n_subj, n_samples)

v_tms_mean_offline_online  = v_tms_subj.mean(axis=1)   # point estimate per subject
v_tms_sd_offline_online    = v_tms_subj.std(axis=1)    # posterior uncertainty per subject

a_tms_fixed = post["a_stim_type_simp_Offline + Online"].values
a_tms_re_z  = post["a_stim_type_simp_Offline + Online|participant"].values
a_tms_re_sd = post["a_stim_type_simp_Offline + Online|participant_sigma"].values
a_tms_re    = a_tms_re_z * a_tms_re_sd[np.newaxis, :]
a_tms_subj_log = a_tms_fixed[np.newaxis, :] + a_tms_re  # log-scale slope
# Option A: keep on log scale (additive effect on log(a)) → correlate directly
a_tms_mean_offline_online = a_tms_subj_log.mean(axis=1)

participant_order = idata.posterior.coords["participant__factor_dim"].values

# Build final DataFrame
df_slopes = pd.DataFrame({
    "participant":      participant_order,
    "v_tms_offline":    v_tms_mean,         # TMS effect on drift rate
    "a_tms_offline":    a_tms_mean_log,     # TMS effect on boundary (log scale)
    "v_tms_offline_online":    v_tms_mean_offline_online,         # TMS effect on drift rate
    "a_tms_offline_online":    a_tms_mean_offline_online,     # TMS effect on boundary (log scale)
})
df_slopes.to_csv("/data/tu_martin_cloud/EXNAT/EXNAT_4_TMS/Results/EXNAT_4_TMS_Analysis/Raw_data/HSSM/HSSM_subject_parameters.csv")

#--------
names = [
    #"v_session_simp_2",
    #"v_session_simp_3",
    "v_stim_type_simp_Offline",
    "v_stim_type_simp_Offline + Online",
]

fig, ax = plt.subplots()

for name in names:
    draws = posterior[name].values.flatten()
    az.plot_kde(draws, ax=ax, label=name)
    hdi = az.hdi(draws, hdi_prob=0.95)
    mean = draws.mean()
    ax.axvline(mean, linestyle="--")
    ax.hlines(0, hdi[0], hdi[1], linewidth=2)

ax.set_xlabel("Effect on drift v")
ax.set_ylabel("Posterior density")
ax.legend()
plt.show()

#------------
idata = models["nback_va_int"]._inference_obj
idata_postpred = models["nback_va_int"].sample_posterior_predictive(
    idata=idata,  # mcmc chains from previous run of model.sample()
    data=df_model,  # empirical p(c) per subject per condition
    inplace=False,
    kind="response"
)

model1_idata = az.extract(models["nback_va_int"].traces)
v_interaction_term = model1_idata['v_stim_type_simp_Offline + Online'].values
bayesian_p = (v_interaction_term < 0).mean()

effect = model1_idata['v_stim_type_simp_Offline:task_simp_2back_single_main_click'].values

mean_eff = effect.mean()
ci_95 = np.quantile(effect, [0.025, 0.975])
p_pos = (effect > 0).mean()
mean_eff, ci_95, p_pos

parameters = list(model1_idata.keys())