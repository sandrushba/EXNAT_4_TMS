import pynibs
import simnibs
import matplotlib
import os
import numpy as np
import glob
import pandas as pd

folder = "/home/ole/Documents/projects/PCD/al/Alina_MA/"
sub_folders = glob.glob(f"{folder}/sub*")
sub_folders.sort()
im_names = ["AG","DLPFC"]
res = {'sub':[],
       'session':[],
       'target':[],
       'idx':[],
       'PCD':[],
       }
for sub in sub_folders:
    sub_short = sub.split("/")[-1]
    # if sub_short != "sub-03":
    #     continue
    print(f"{sub_short}")
    # read InstrumentMarker files
    im = glob.glob(f"{sub}/InstrumentMarker/*.xml")
    assert len(im) == 1
    im = im[0]
    im_arr, im_names, im_times = pynibs.expio.localite.get_marker(im, markertype="InstrumentMarker")
    print(f"\t{im_names}")

    # read triggermarker
    tms = glob.glob(f"{sub}/*.xml")
    for idx, tm in enumerate(tms):

        if "offline_online_offline" in tm:
            # get IM
            # dual site
            session = "offline_online"
            target = 'AG'
            idx_tm = im_names.index(target)
            arr_tm = im_arr[idx_tm]
        elif 'offline.xml' in tm:
            # single site
            session = "offline"
            target = 'AG'
            if sub_short == "sub-03":
                target = "AG Reverse"
            idx_tm = im_names.index(target)
            arr_tm = im_arr[idx_tm]
        elif 'offline_online_online.xml' in tm:
            session = "offline_online"
            target = 'DLPFC'
            idx_tm = im_names.index(target)
            arr_tm = im_arr[idx_tm]
        else:
            print(f"Cannot find session for {tm}")
            continue
        im_arrs = pynibs.expio.localite.read_triggermarker(tm)[0]
        print(f"\t{session: <16} - {target: <6}")
        # get the InstrumentMarker for AG from correct file

        # calculate absolute and relative coil displacements
        delta_pos_abs, delta_rot_abs, delta_pos_rel, delta_rot_rel = pynibs.util.quality_measures.calc_tms_motion_params(im_arrs,
                                                                                                   reference=arr_tm)

        # compute PCD
        if target == "AG Reverse":
            target = "AG"
        pcd, delta_pos, delta_rot = pynibs.util.quality_measures.compute_pcd(delta_pos_abs, delta_rot_abs)
        for idx_pcd, pcd_val in enumerate(pcd):
            res["sub"].append(sub_short)
            res["session"].append(session)
            res["target"].append(target)
            res["idx"].append(idx_pcd)
            res["PCD"].append(pcd_val)

res_pd = pd.DataFrame().from_dict(res)
mask = res_pd["PCD"].isna()
res_pd["PCD"] = res_pd["PCD"].interpolate()
res_pd['interpolated'] = mask.astype(int)

res_pd.to_csv(f"{folder}/pcd_combined.csv", index=False)
quit()


# get triggermarker file
tm_fn = "/data/p_03049/Alina_MA/sub-17/TriggerMarkers_Coil1_20250620131939185_offline.xml"
assert os.path.exists(tm_fn)

# this can also return the time of each pulse in case you want to look at that (for offline vs online)
im_arrs = pynibs.read_triggermarker_localite(tm_fn)[0]

# get the InstrumentMarker for AG from correct file


# calculate absolute and relative coil displacements
delta_pos_abs, delta_rot_abs, delta_pos_rel, delta_rot_rel = pynibs.calc_tms_motion_params(mats, reference=ag_arr)

# compute PCD
pcd, delta_pos, delta_rot = pynibs.compute_pcd(delta_pos_abs, delta_rot_abs)
pcd.max()

# plot PCD
axes = pynibs.plot_tms_motion_parameter(delta_pos_rel, delta_rot_rel, pcd)
matplotlib.pyplot.show()