import os
import concurrent.futures

# Define paths
base_dir = "/data/p_03049/MRI_TMS_Data/"
charm_dir = "/data/u_martin_software/miniforge3/envs/simnibs_env/bin/charm"

def create_head_mesh(id, t1_file, sub_dir, charm_dir, fs_dir, t2_file=None):
    if t2_file is not None:
        cmd = f"python {charm_dir} {id} {t1_file} {t2_file} --forceqform --fs-dir {fs_dir} --forcerun"
        print("Running command:", cmd)
        os.chdir(sub_dir)
        os.system(cmd)
    else:
        cmd = f"python {charm_dir} {id} {t1_file} --forceqform --fs-dir {fs_dir} --forcerun"
        print("Running command:", cmd)
        os.chdir(sub_dir)
        os.system(cmd)

def create_head_mesh_wrapper(sub):
    sub_dir = os.path.join(base_dir, sub)
    t1_file = os.path.join(sub_dir, f"{sub}_biasCorrected_T1.nii")
    t2_file = os.path.join(sub_dir, f"{sub}_T2_LTA_registered.nii.gz")
    fs_dir = os.path.join(sub_dir, "freesurfer_recon", sub)
    if os.path.isfile(t2_file):
        create_head_mesh(sub, t1_file, sub_dir, charm_dir, fs_dir, t2_file)
    else:
        print("Subject was already created")
        create_head_mesh(sub, t1_file, sub_dir, charm_dir, fs_dir)

subs = [d for d in os.listdir(base_dir) if "sub-" in d]

with concurrent.futures.ProcessPoolExecutor() as executor:
    futures = []
    for sub in subs:
        #t2_file = os.path.join(base_dir, sub, f"{sub}_T2.nii.gz")
        sub = "sub-06"
        print(sub)
        print("meshing started")
        future = executor.submit(create_head_mesh_wrapper, sub)
        futures.append(future)
        # if os.path.isfile(t2_file):
        #     print(sub)
        #     print("meshing started with T2 file")
        #     future = executor.submit(create_head_mesh_wrapper, sub)
        #     futures.append(future)
        # else:
        #     print(sub)
        #     print("meshing started without T2 file")
        #     future = executor.submit(create_head_mesh_wrapper, sub)
        #     futures.append(future)

    for future in concurrent.futures.as_completed(futures):
        future.result()