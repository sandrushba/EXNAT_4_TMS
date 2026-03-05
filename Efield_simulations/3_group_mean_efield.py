import os
import pandas as pd
import numpy as np
from simnibs import localite, read_msh
import gmsh

# Define paths
base_dir = "/data/p_03049/MRI_TMS_Data/"
target_dir_efields = "/data/p_03049/Results/E-field_simulations"

# Define list of subjects
subs = os.listdir(base_dir)

# Read file with subject and session info
stim_dir = "/data/p_03049/Scripts/TMS_simulation/EXNAT_4_TMS_session_info.txt"
stim_intensity = pd.read_csv(stim_dir, sep="\t")

# Define n of sessions
sessions = ("offline", "offline_online")  # "sham"

# Load simulation results
results_folder = "efield_sim_{0}/fsavg_overlays"
fsavg_msh_name = "{0}_TMS_1-000{1}_MagVenture_MCF-B65_new_scalar_fsavg.msh"
field_name = 'E_magn'

fields = {'offline': {'AG': []}, 'online': {'AG': [], 'DLPFC': []}}

for sess in ['offline', 'offline_online']:
    for sub in subs:
        if "sub-" not in sub:
            continue
        print(sub, "and", sess)
        sub_dir = os.path.join(base_dir, sub)

        marker_col = sess + "_marker_file"
        marker_file = stim_intensity.loc[stim_intensity.participant == sub, marker_col].iloc[0]
        tms_list = localite().read(marker_file)

        if sess == "offline":
            target_list = [i for i in tms_list.pos if i.name == "AG"]
        else:  # offline_online
            target_list = [i for i in tms_list.pos if i.name in ("AG", "DLPFC")]

        for idx, target in enumerate(target_list):
            mesh_idx = idx + 1
            msh_path = os.path.join(
                sub_dir,
                results_folder.format(sess),
                fsavg_msh_name.format(sub, mesh_idx)
            )

            # Read mesh and extract subject's own efield
            results_fsavg = read_msh(msh_path)

            # Append efield values to dict
            sess_key = 'offline' if target.name == 'AG' and sess == 'offline' else 'online'
            fields[sess_key][target.name].append(results_fsavg.field[field_name].value)

            #results_fsavg.write(os.path.join(target_dir_efields, f"{sess_key}_{target}.msh")) --> not sure this is the correct way to save a mesh

# group_fields = pd.DataFrame(fields, columns=["sub", "sess", "site", "Emagn"])
# group_fields.to_csv(os.path.join(base_dir, "group_fields.txt"), index = False, sep = "\t")

## Calculate and plot averages
for session in fields:
    for target in fields[session]:
        mesh_fields = np.vstack(fields[session][target])
        avg_field = np.mean(mesh_fields, axis=0)
        std_field = np.std(mesh_fields, axis=0)

        results_fsavg.nodedata = []  # cleanup fields
        # Add node fields for the average and std deviation
        results_fsavg.add_node_field(avg_field, 'E_magn_avg')
        results_fsavg.add_node_field(std_field, 'E_magn_std')

        print(f"Showing for {session} - {target}")
        results_fsavg.view(visible_fields='E_magn_avg').show()



# for sub in subs:
#     if "sub-" in sub:
#         # read mesh with results transformed to fsaverage space
#         results_fsavg = simnibs.read_msh(
#             os.path.join(base_dir, sub, results_folder, fsavg_msh_name.format(sub))
#         )
#         # save the field in each subject
#         fields.append(results_fsavg.field[field_name].value)
#
# ## Calculate and plot averages
# # Calculate
# fields = np.vstack(fields)
# avg_field = np.mean(fields, axis=0)
# std_field = np.std(fields, axis=0)
#
# # Plot
# results_fsavg.nodedata = [] # cleanup fields
# results_fsavg.add_node_field(avg_field, 'E_magn_avg') # add average field
# results_fsavg.add_node_field(std_field, 'E_magn_std') # add std field
#
# # show surface with the fields
# results_fsavg.view(visible_fields='E_magn_avg').show()

## Calculate average in an ROI defined using an atlas
# load atlas and define a region
# atlas = simnibs.get_atlas('HCP_MMP1')
# region_name = 'lh.4'
# roi = atlas[region_name]
# # visualize region
# results_fsavg.add_node_field(roi, region_name)
# results_fsavg.view(visible_fields=region_name).show()
#
# # calculate mean field using a weighted mean
# node_areas = results_fsavg.nodes_areas()
# avg_field_roi = np.average(avg_field[roi], weights=node_areas[roi])
# print(f'Average {field_name} in {region_name}: ', avg_field_roi)
# results_fsavg.add_node_field(roi, region_name)