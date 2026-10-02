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
    "dual_task_baseline": "dual_task_baseline",
    "dual_task_v_full_red": "dual_task_v_full_red",
    "dual_task_v_full_slopes": "dual_task_v_full_slopes",
    "dual_task_va": "dual_task_va",
    "dual_task_va_slopes": "dual_task_va_slopes",
    "dual_task_vt": "dual_task_vt",
    "dual_task_vt_slopes": "dual_task_vt_slopes",
    "dual_task_vz_slopes": "dual_task_vz_slopes",
    "dual_task_vat": "dual_task_vat",
}

models = load_models(model_paths)

#  ---------------------------------------------------
# MODEL COMPARISON
#  ---------------------------------------------------

model_comparison = az.compare(
    {
        "Model 1": models["dual_task_baseline"].traces, # model without any specification
        #"Model 2": models["dual_task_v_full_red"].traces, # model only with fixed and random effects for v
        "Model 3": models["dual_task_v_full_slopes"].traces, # model with full structure for v and random participant-wise intercepts for a,t,z
        #"Model 4": models["dual_task_va"].traces # model with full structure for v and full fixed effects and random participant-wise intercepts for a
        "Model 5": models["dual_task_va_slopes"].traces, # model with full structure for v and a (incl random slopes) ***
        #"Model 6": models["dual_task_vt"].traces, # model with full structure for v and full fixed effects and random participant-wise intercepts for t
        "Model 7": models["dual_task_vt_slopes"].traces, # model with full structure for v and t (incl random slopes)
        "Model 8": models["dual_task_vz_slopes"].traces, # model with full structure for v and t (incl random slopes)
        "Model 9": models["dual_task_vat"].traces, # model with full structure for v and full fixed effects and random participant-wise intercepts for a and t
    },
    ic="loo",
)

model_comparison.to_csv("/data/p_03049/EXNAT_4_TMS/HSSM/model_comparison_dual_task_loo.csv")

loo1 = az.loo(models["dual_task_va_slopes"].traces)
loo2 = az.loo(models["dual_task_vt_slopes"].traces)
loo3 = az.loo(models["dual_task_vat"].traces)

# if you need to compute log-likelihood post-hoc for a model
idata = pm.compute_log_likelihood(
    models['dual_task_v_full_slopes'].traces,
    model=models['dual_task_v_full_slopes'].pymc_model,
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

## generate model summaries and save them in html format
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


#  ---------------------------------------------------
# PLOTTING
#  ---------------------------------------------------

# generate graph of winning model
graph = models["dual_task_va_slopes"].graph(name="Dual_task_va_slopes")
graph.view()

# plot posterior predictions of best models
hssm.plotting.plot_predictive(models["dual_task_va_slopes"])
plt.savefig(os.path.join(plot_dir, "ppc_dual_task_va_slopes.png"), dpi=200)
plt.show()

# plot posterior predictions of best models for each participant
hssm.plotting.plot_predictive(models["dual_task_va_slopes"], col="participant", col_wrap=5)
plt.savefig(os.path.join(plot_dir, "ppc_dual_task_va_slopes_participants.png"), dpi=200)
plt.show()

###############
# Specify a plot with one row and two columns
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
# Plot the posterior samples for condition 1
az.plot_posterior(
    models["nback_va"].traces.posterior, var_names=('v_stim_type_simp_Offline:task_simp_2back_single_main_click'),
    ax=ax, color="blue", hdi_prob=0.95
)

# Plot the posterior samples for condition 2
az.plot_posterior(models["nback_va"].traces.posterior, var_names=('v_stim_type_simp_Offline + Online:task_simp_2back_single_main_click'),
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
#-------
posterior = models["dual_task_va_slopes"].traces.posterior

all_params = list(posterior.data_vars)
az.plot_posterior(posterior, var_names=all_params)
plt.show()

group_vars = [var for var in posterior.data_vars if "participant" not in var]
df = az.extract(models["dual_task_va_slopes"].traces, group="posterior", var_names=group_vars).to_dataframe()

output_dir = "/data/p_03049/EXNAT_4_TMS/HSSM/Model_summaries"
df.to_csv(os.path.join(output_dir, "dual_task_va_int.csv"))

hssm.plotting.plot_model_cartoon(models["dual_task_va_slopes"],
                                 n_samples=10,
                                 bins=20,
                                 plot_predictive_mean=True,
                                 plot_predictive_samples=True,
                                 n_trajectories=2)
plt.show()

#-------- extract subject-specific params ----
posterior = models["dual_task_va_slopes"].traces.posterior

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
df.to_csv(os.path.join(output_dir, "dual_task_va_int_subject_params.csv"))

#----- plot individual subjects -----
hssm.plotting.plot_predictive(models["nback_v_full_slopes"])
plt.savefig(os.path.join(plot_dir, "ppc_nback_full_slopes.png"), dpi=150)

hssm.plotting.plot_predictive(models["nback_va_int"], col="participant", col_wrap=5)
plt.savefig(os.path.join(plot_dir, "ppc_nback_full_slopes_participants.png"), dpi=150)

az.plot_posterior(model_traces["nback_va"], var_names=('v_stim_type_simp_Offline:task_simp_2back_single_main_click'))
plt.show()

az.plot_posterior(model_traces["nback_va"], var_names=('a_task_simp_2back_single_main_click'))
plt.show()

az.plot_forest(model_traces["nback_va"], var_names=('a_stim_type_simp_Offline:task_simp_2back_single_main_click'),
               combined=False) # show i
