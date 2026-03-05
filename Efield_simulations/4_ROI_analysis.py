import os
import numpy as np
import pandas as pd
import ast
from simnibs import read_msh, mni2subject_coords, localite

# Define paths
base_dir = "/data/p_03049/MRI_TMS_Data/"

# Define n of sessions
sessions = ("offline", "offline_online")  # "sham"

# Read file with subject and session info
stim_dir = "/data/p_03049/Scripts/TMS_simulation/EXNAT_4_TMS_session_info.txt"
stim_intensity = pd.read_csv(stim_dir, sep="\t")

# Read file with subjects with individual stim coordinates (based on EXNAT_3)
indiv_coordinates_dir = "/data/p_03049/Scripts/TMS_simulation/EXANT_4_TMS_indiv_coordinates.csv"
indiv_coordinates = pd.read_csv(indiv_coordinates_dir, sep=",")

# Load simulation results
m2m_dir = "m2m_{0}"
results_folder = "efield_sim_{0}"
msh_name = "{0}_TMS_1-000{1}_MagVenture_MCF-B65_new_scalar.msh"

# Coordinates for ROI creation
coords = [([-60.5, -54.5, 15.5], "AG"), ([-46.5, 29.5, 25.5], "DLPFC")]
coords = pd.DataFrame(coords, columns=["coord", "site"]).set_index('site').T

# Define list of subjects
subs = os.listdir(base_dir)

def get_subject_coords(subs, indiv_coordinates, coords, base_dir):
    dict_coords = {}
    for sess in ['offline', 'offline_online']:
        for sub in subs:
            if "sub-" not in sub:
                continue
            # Check indivdf first
            sub_indiv_coord = indiv_coordinates[indiv_coordinates['participant'] == sub]
            if not sub_indiv_coord.empty:
                used_df = sub_indiv_coord
                source = 'indiv_coordinates'
            else:
                used_df = coords
                source = 'coords'

            roi_coords = {}
            for roi in ['AG', 'DLPFC']:
                if roi in used_df.columns:
                    value = used_df.iloc[0][roi]
                    # If it's a string, convert to list
                    if isinstance(value, str):
                        try:
                            value = ast.literal_eval(value)
                        except Exception:
                            raise ValueError(f"Malformed coordinate string for {roi}: {value}")
                    roi_coords[roi] = value
            dict_coords[(sub, sess)] = {'coords': roi_coords, 'source': source}
    return dict_coords

def calculate_efield(coord, head_mesh, m2m_folder):

    # Crop the mesh so we only have gray matter volume elements (tag 2 in the mesh)
    gray_matter = head_mesh.crop_mesh(2)

    # Define the ROI
    sub_coords = mni2subject_coords(coord, m2m_folder)
    # define sphere radius
    r = 10.
    #r = 5.

    # Electric fields are defined in the center of the elements
    # get element centers
    elm_centers = gray_matter.elements_baricenters()[:]
    # determine the elements in the ROI
    roi = np.linalg.norm(elm_centers - sub_coords, axis=1) < r
    # get the element volumes, we will use those for averaging
    elm_vols = gray_matter.elements_volumes_and_areas()[:]

    ## Plot the ROI
    # gray_matter.add_element_field(roi, 'roi')
    # gray_matter.view(visible_fields='roi').show()

    ## Get field and calculate the mean
    # get the field of interest
    field_name = 'magnE'
    field = gray_matter.field[field_name][:]

    # Calculate the mean
    # mean_magnE = np.average(field[roi], weights=elm_vols[roi])
    # Alternative: calculate the top % strongest values in our ROI
    mean_magnE = np.percentile(field[roi], 95)
    print('mean ', field_name, ' in ROI ', target.name, " in session", sess, ' : ', mean_magnE)
    return mean_magnE

target_coords = get_subject_coords(subs, indiv_coordinates, coords, base_dir)
average_fields = pd.DataFrame()

for sess in ['offline', 'offline_online']:
    for sub in subs:
        if "sub-" not in sub:
            continue
        print(sub, "and", sess)
        sub_dir = os.path.join(base_dir, sub)

        roi_coord_dict = target_coords.get((sub, sess), {}).get('coords', {})
        if not roi_coord_dict:
            raise ValueError(f"No coordinates found for subject {sub} in session {sess}")

        # Find the correct marker file for the session/subject
        marker_col = sess + "_marker_file"
        marker_file = stim_intensity.loc[stim_intensity.participant == sub, marker_col].iloc[0]
        tms_list = localite().read(marker_file)
        if sess == 'offline':
            target_list = [i for i in tms_list.pos if i.name == "AG"]
        else:  # offline_online
            # The order in the marker file determines mesh order!
            target_list = [i for i in tms_list.pos if i.name in ("AG", "DLPFC")]

        for idx, target in enumerate(target_list):
            mesh_idx = idx + 1
            roi_name = target.name
            coord = roi_coord_dict.get(roi_name)
            if coord is None:
                continue
            msh_path = os.path.join(
                sub_dir,
                results_folder.format(sess),
                msh_name.format(sub, mesh_idx)
            )
            head_mesh = read_msh(msh_path)
            m2m_folder = os.path.join(sub_dir, m2m_dir.format(sub))
            mean_field = calculate_efield(coord, head_mesh, m2m_folder)
            temp = pd.DataFrame({
                "participant": [sub],
                "session": [sess],
                "roi": [roi_name],
                "mean_field": [mean_field]
            })
            average_fields = pd.concat([average_fields, temp])

average_fields = average_fields.sort_values(by="participant")
average_fields.to_csv(os.path.join(base_dir, "average_efields_sphere_10mm_percentile_95.txt"), index=False, sep="\t")
