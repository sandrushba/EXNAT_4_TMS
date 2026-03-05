import os
import concurrent.futures

# Define base directory
base_dir = "/data/p_03049/MRI_TMS_Data/"

def register_t2_to_t1(sub, base_dir):
    sub_dir = os.path.join(base_dir, sub)
    t1_file = os.path.join(sub_dir, f"{sub}_biasCorrected_T1.nii")
    t2_file = os.path.join(base_dir, sub, f"{sub}_T2.nii.gz")
    output_file_lta = os.path.join(base_dir, sub, f"{sub}_T2_LTA.lta")
    output_registered_t2 = os.path.join(base_dir, sub, f"{sub}_T2_LTA_registered.nii.gz")
    # Check if realignment is needed (e.g., output doesn't exist)
    if os.path.isfile(t2_file) and not os.path.isfile(output_registered_t2):
        cmd = f"FREESURFER mri_robust_register --mov {t2_file} --dst {t1_file} --lta {output_file_lta} --mapmov {output_registered_t2} --satit"
        print(f"Running: {cmd}")
        os.system(cmd)
    return output_registered_t2 if os.path.isfile(output_registered_t2) else None

def run_recon_all(sub, base_dir):
    sub_dir = os.path.join(base_dir, sub)
    output_dir = os.path.join(base_dir, sub, "freesurfer_recon")
    os.makedirs(output_dir, exist_ok=True)
    t1_file = os.path.join(sub_dir, f"{sub}_biasCorrected_T1.nii")
    t2_registered_file = os.path.join(base_dir, sub, f"{sub}_T2_LTA_registered.nii.gz")
    t2_file = t2_registered_file if os.path.isfile(t2_registered_file) else None
    if t2_file:
        cmd = f"FREESURFER recon-all -subject {sub} -i {t1_file} -T2 {t2_file} -T2pial -all -sd {output_dir}"
    else:
        cmd = f"FREESURFER recon-all -subject {sub} -i {t1_file} -all -sd {output_dir}"
    print(f"Running: {cmd}")
    os.system(cmd)

if __name__ == "__main__":
    # List all subject directories
    subs = [d for d in os.listdir(base_dir) if d.startswith("sub-")]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for sub in subs:
            t1_file = os.path.join(base_dir, sub, f"{sub}_biasCorrected_T1.nii")
            if os.path.isfile(t1_file):
                print(f"Registering and submitting {sub} for recon-all.")
                register_t2_to_t1(sub, base_dir)  # Do registration step synchronously or in prior job
                future = executor.submit(run_recon_all, sub, base_dir)
                futures.append(future)
            else:
                print(f"Skipping {sub}: No T1 image found.")
        for future in concurrent.futures.as_completed(futures):
            future.result()