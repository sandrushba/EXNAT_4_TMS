import os
import pandas as pd
import numpy as np
from simnibs import localite, transformations
import nibabel as nib
import meshio

# Define paths
base_dir = "/data/p_03049/MRI_TMS_Data/"
target_dir_efields = "/data/p_03049/Results/E-field_simulations"

# Define list of subjects
subs = os.listdir(base_dir)
subs.sort()

# Read file with subject and session info
stim_intensity = pd.read_csv("/data/p_03049/EXNAT_4_TMS/Efield_simulations/EXNAT_4_TMS_session_info.txt", sep="\t")

# Define n of sessions
sessions = ("offline", "offline_online")  # "sham"

# Load simulation results
depth = 0.99
# depth = 0.5
# results_folder = "efield_sim_{0}/fsavg_overlays"
fsavg_msh_name = "{0}_TMS_1-000{1}_MagVenture_MCF-B65_new_scalar_fsavg.E.magn"
field_name = 'E_magn'

fields = {'offline':
              {'AG': []},
          'online':
              {'AG':[],
               'DLPFC': []
               }
          }

# which surface for plotting - pial, inflated, etc.
surf = 'pial'
surf = 'inflated0030'
surf = 'inflated0015'
surf = 'pial_inflated0015'

output_folder = "/data/p_03049/Results/E-field_simulations/fsaverage"
fs_average_subject_folder = "/data/p_03049/EXNAT_4_TMS/Efield_simulations/fsavg_geometry/"
for hem in ['lh','rh']:
    coords, faces = nib.freesurfer.read_geometry(f"{fs_average_subject_folder}/{hem}.{surf}")
    for sess in ['offline', 'offline_online']:
        for sub in subs:
            if "sub-" not in sub:
                continue
            print(sub, "and", sess)
            sub_dir = os.path.join(base_dir, sub)

            marker_col = sess + "_marker_file"
            marker_file = stim_intensity.loc[stim_intensity.participant == sub, marker_col].iloc[0]
            try:
                tms_list = localite().read(marker_file)
            except AssertionError as e:
                print(e)
                continue

            if sess == "offline":
                target_list = [i for i in tms_list.pos if i.name == "AG"]
            else:  # offline_online
                target_list = [i for i in tms_list.pos if i.name in ("AG", "DLPFC")]

            for idx, target in enumerate(target_list):
                out_folder = f"{sub_dir}/efield_sim_{sess}/subject_overlays_{depth}/"
                out_fsavg = f"{sub_dir}/efield_sim_{sess}/fsavg_overlays_{depth}/"
                mesh_idx = idx + 1
                msh_path = (f"{sub_dir}/"
                            f"efield_sim_{sess}/fsavg_overlays_{depth}/"
                            f"{hem}.{sub}_TMS_1-000{mesh_idx}_MagVenture_MCF-B65_new_scalar.fsavg.E.magn")
                if not os.path.exists(out_fsavg) or not os.path.exists(msh_path):
                    print(f"\tCreating fsaverage with depth = {depth}")
                    subpath = f"/data/p_03049/MRI_TMS_Data/{sub}/m2m_{sub}"

                    f = f"/data/p_03049/MRI_TMS_Data/{sub}/efield_sim_{sess}/{sub}_TMS_1-000{mesh_idx}_MagVenture_MCF-B65_new_scalar.msh"
                    f_geo = f.replace("scalar.msh","coil_pos.geo")
                    transformations.middle_gm_interpolation(
                            f, subpath, out_folder,
                            out_fsaverage=out_fsavg, depth=depth,
                            open_in_gmsh=False, f_geo=f_geo)



                # read the morph data
                e = nib.freesurfer.read_morph_data(msh_path)

                # Append efield values to dict
                sess_key = 'offline' if target.name == 'AG' and sess == 'offline' else 'online'
                fields[sess_key][target.name].append(e)


    # group_fields = pd.DataFrame(fields, columns=["sub", "sess", "site", "Emagn"])
    # group_fields.to_csv(os.path.join(base_dir, "group_fields.txt"), index = False, sep = "\t")

    ## Calculate and plot averages
    for session in fields:
        for target in fields[session]:
            mesh_fields = np.vstack(fields[session][target])
            print(f"{session} - {target}: Found {mesh_fields.shape[0]} fields.")
            avg_field = np.mean(mesh_fields, axis=0)
            std_field = np.std(mesh_fields, axis=0)

            # here we put the data for each .vtk file into one dict:
            point_data = {'e_avg': avg_field, 'e_std': std_field}

            output_fn = f"{output_folder}/{session}_{target}_{hem}_{surf}_{depth}.vtk"
            print(f"Writing {output_fn}")
            meshio.Mesh(np.squeeze(coords), [('triangle', faces)], point_data=point_data).write(output_fn)
