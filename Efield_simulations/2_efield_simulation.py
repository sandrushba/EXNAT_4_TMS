import os
# from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
# import glob
# import datetime
from simnibs import sim_struct, run_simnibs, localite

# Define paths
base_dir = "/data/p_03049/MRI_TMS_Data/"

# Define coil model
TMS_coil = "/data/p_03049/Scripts/TMS_simulation/MagVenture_MCF-B65_new.ccd"
#TMS_coil = "/data/p_02623/Scripts/TMS_simulation/MagVenture_MCF-B65.nii.gz"
setup_factor = 1.43

# Read file with subject and session info
stim_dir = "/data/p_03049/Scripts/TMS_simulation/EXNAT_4_TMS_session_info.txt"
stim_intensity = pd.read_csv(stim_dir, sep="\t")

# Define subjects
subs = os.listdir(base_dir)

# Define n of sessions
sessions = ("offline", "offline_online")  # "sham"

# Define file formats
# anatomy_form = "{0}_biasCorrected_T1.nii"
mesh_form = "m2m_{0}"

# Loop over subjects to start
for sub in subs:
    if "sub-" in sub:
        print(sub)
        sub_dir = os.path.join(base_dir, sub)
        # determine type of stimulation and session
        for sess in sessions:
            s = sim_struct.SESSION()

            print(sess)
            #sess_date = stim_intensity.loc[stim_intensity.]

            # extract data from instrument marker xml file
            '''alternative approach to find correct instrument marker file when absolute path of marker file is not given'''
            # for column in stim_intensity:
            #     if column == sess:
            #         sess_date = stim_intensity.loc[stim_intensity.participant == int(sub.removeprefix("sub-")), column].iloc[0]
            # first convert sess date to correct format
            # date_obj = datetime.datetime.strptime(sess_date, "%d/%m/%Y")
            # sess_date = date_obj.strftime("%Y%m%d")
            #
            # sess_dir = glob.glob(os.path.join(base_dir, sub, "sub-*", "Sessions", "Session_" + str(sess_date) + "*", "InstrumentMarkers"))
            # markers = os.listdir(sess_dir[0])
            # markers.sort()
            # curr_marker = len(markers)-1
            # tms_list = localite().read(os.path.join(sess_dir[0], markers[curr_marker]))

            output_dir = os.path.join(sub_dir, "efield_sim_" + sess)
            target_check = os.path.join(sub_dir, "efield_sim_" + sess)  # The target folder

            if not os.path.exists(target_check):
                if sess == "offline":
                    for column in stim_intensity:
                        #if sess in column:
                        if column == sess + "_marker_file":
                            curr_marker = stim_intensity.loc[stim_intensity.participant == sub, column].iloc[0]
                elif sess == "offline_online":
                    for column in stim_intensity:
                        if column == sess + "_marker_file":
                            curr_marker = stim_intensity.loc[stim_intensity.participant == sub, column].iloc[0]

                tms_list = localite().read(curr_marker)
                if sess == "offline":
                    tms_list.pos = [i for i in tms_list.pos if i.name == "AG"]
                elif sess == "offline_online":
                    tms_list.pos = [i for i in tms_list.pos if i.name == "AG" or i.name == "DLPFC"]

                # pos_names = []
                # for curr_pos in range(len(tms_list.pos)):
                #     pos_names.append(tms_list.pos[curr_pos].name)

                # define name of coil
                tms_list.fnamecoil = TMS_coil

                # add instrument marker information to session object
                s.add_tmslist(tms_list)

                # calculate dI/dT value based on stimulation intensity and set-up factor
                ## stim intensity is in %, didt in A/us
                if sess == "offline":
                    MSO = stim_intensity.loc[stim_intensity.participant == sub, "intensity_offline"].iloc[0]
                    didt = MSO * setup_factor
                    print("Field strength in A/us is", didt)
                    #didt_s = didt * 1e6
                    s.poslists[0].pos[0].didt = didt * 1e6
                elif sess == "offline_online":
                    for pos_item in s.poslists[0].pos:
                        # Read the name field to determine which intensity to assign
                        if pos_item.name == "AG":
                            MSO = stim_intensity.loc[stim_intensity.participant == sub, "intensity_offline"].iloc[0]
                        elif pos_item.name == "DLPFC":
                            MSO = stim_intensity.loc[stim_intensity.participant == sub, "intensity_online"].iloc[0]
                        else:
                            raise ValueError(f"Unknown target name: {pos_item.name}")
                        didt = MSO * setup_factor
                        print(f"Field strength for {pos_item.name} in A/us is {didt}")
                        pos_item.didt = didt * 1e6

                # define path for mesh
                mesh_dir = os.path.join(sub_dir, mesh_form.format(sub))
                s.subpath = mesh_dir

                # define output folder
                output_dir = os.path.join(sub_dir, "efield_sim_" + sess)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                s.pathfem = output_dir

                # open results directly in gmesh? (default = yes)
                s.open_in_gmsh = False

                # map results on surface
                s.map_to_surf = True

                # map results to fsaverage
                s.map_to_fsavg = True

                # map results to volume
                #s.map_to_vol = True

                print("This is subject", sub)
                run_simnibs(s)

            else:
                print(f"{target_check} already exists. Skipping simulation.")