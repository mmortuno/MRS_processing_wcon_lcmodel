#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRS Processor
Improved and generalized for any number of VOIs.
Author: Maria Ortuño (refactored version)
"""

import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import suspect
from nipype.interfaces import fsl
import nipype.interfaces.spm as spm
from nipype.interfaces.spm import SPMCommand
import subprocess
import warnings
warnings.filterwarnings('ignore', 
    message="The value length.*exceeds the maximum length.*",
    module="pydicom"
)

# ----------------------------
# Configuration and Constants
# ----------------------------

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Set SPM MATLAB command
SPMCommand.set_mlab_paths(
    matlab_cmd='/usr/local/MATLAB/R2023b/bin/matlab -nodesktop -nosplash',
    use_mcr=False
)

# Add Path to heudiconv heuristic file
heuristic_path = ''

# Base directories (change these as needed)
MRS_DATA_DIR = Path('')

# Define the list of base VOIs; to add more VOIs, simply update this list.
VOI_LIST = ['dmPF', 'mTL']

# LCModel parameters (if needed by spectral fitting)
LCMODEL_PARAMS = {
    # Add path to basis set
    "FILBAS": "",
    "LCSV": True,
    "LTABLE": True
}

#Change if needed
echot = "30"

# ----------------------------
# Helper Functions
# ----------------------------

protocol_dict = {"svs_se_ ACC FRONTAL LATERAL IZQ": "dmPF", 
                 "svs_se_ ACC_h2o FRONTAL LATERAL IZQ": "dmPF_H2O",
                 "svs_se_ HIPOCAMPO IZQUIERDO": "mTL",
                 "svs_se_ h2o_HIPOCAMPO IZQUIERDO": "mTL_H2O"}


def is_dicom(file_path):
    """
    Checks if the file at file_path is a DICOM file by reading its header.
    Returns True if it finds the 'DICM' magic word at byte 128.
    """
    try:
        with open(file_path, 'rb') as f:
            f.seek(128)
            if f.read(4) == b'DICM':
                return True
    except Exception as e:
        logging.warning(f"Could not check if file {file_path} is DICOM: {e}")
    return False

def process_mrs_directory(root_dir, protocol_dict, subject_name, timepoint, heuristic_path):
    """
    Assess the folder structure in root_dir, convert DICOM files to NIfTI BIDS using heudiconv,
    and process RDA files based on a protocol dictionary.

    Parameters:
        root_dir (str): The directory to assess.
        protocol_dict (dict): Dictionary mapping protocols to target names (e.g., {"protocol1": "dmPF", "protocol2": "mTL"}).
        subject_name (str): Subject identifier.
        timepoint (str): Timepoint identifier.
        heuristic_path (str): Path to the heuristic file for heudiconv.
    """
    subject_folder = os.path.join(root_dir, subject_name)
    
    rda_files = []
    dicom_files = []
    subdirs = []

    # Walk the directory tree and categorize files
    for dirpath, dirnames, filenames in os.walk(subject_folder):
        for d in dirnames:
            subdirs.append(os.path.join(dirpath, d))

        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if filename.lower().endswith('.rda'):
                rda_files.append(filepath)
            # Use your custom is_dicom() function to check if the file is a DICOM file.
            elif is_dicom(filepath):
                dicom_files.append(filepath)

    # Report findings
    unique_dirs = {os.path.dirname(f) for f in dicom_files}
    print(f"Found {len(subdirs)} directories, {len(rda_files)} RDA files, and {len(dicom_files)} DICOM files across {len(unique_dirs)} folder(s).")

    # ---------------------------------------------------
    # 1) Process all DICOM files with heudiconv (if any)
    # ---------------------------------------------------
    if dicom_files:
        print("DICOM files detected. Converting to NIfTI BIDS using heudiconv...")

        # Output directory for anatomical data (destination for NIfTI files)
        anat_output_dir = os.path.join(subject_folder, "heudiconv")
        os.makedirs(anat_output_dir, exist_ok=True)

        # Instead of using "-d" (which can fail if we have non-DICOM files/folders),
        # we pass each directory containing DICOMs via "--files".
        heudiconv_cmd = [
            "heudiconv",
            "-s", subject_name,            # subject identifier
            "-o", anat_output_dir,         # output directory
            "-ss", timepoint,         # session
            "-f", heuristic_path,          # your heuristic file path
            "--bids"                       # produce BIDS output
        ]

        # Add each unique directory with --files
        for d in sorted(unique_dirs):
            heudiconv_cmd.extend(["--files", d])

        print("Running heudiconv command:")
        print(" ".join(heudiconv_cmd))
        subprocess.run(heudiconv_cmd, check=True)
    
    subject_folder = Path(subject_folder)

    for path in subject_folder.rglob('*'):
        if path.is_dir() and (not any(parent.name.startswith('.') for parent in path.parents)) and path.name.startswith('ses-'):
            
            print(f"Found anat folder at: {path}")
            
            session = str.split(str(path), '/')[-1] 

            shutil.copytree(path, subject_folder / session, dirs_exist_ok=True)
            
            print(f"Copied '{path}' to '{anat_output_dir}'")
            break
    else:
        # If the loop completes without a break, no anat folder was found
        print(f"No 'anat' folder found in {subject_folder}")

    # ---------------------------------------------------
    # 2) Process RDA files (if any)
    # ---------------------------------------------------
    if rda_files:
        print("RDA files detected. Processing according to protocol dictionary...")

        for rda_file in rda_files:
            protocol = None
            # Example: Read the file and extract the protocol (adjust as needed).
            try:
                with open(rda_file, 'r', errors='ignore') as f:
                    for line in f:
                        if "ProtocolName:" in line:
                            protocol = line.split("ProtocolName: ")[-1].strip()
                            break
            except Exception as e:
                print(f"Error reading {rda_file}: {e}")
                continue

            if protocol is None:
                print(f"Protocol not found in {rda_file}; skipping this file.")
                continue

            if protocol not in protocol_dict:
                print(f"Protocol '{protocol}' not found in the provided protocol dictionary; skipping {rda_file}.")
                continue

            target_protocol = protocol_dict[protocol]

            target_dir = os.path.join(root_dir, subject_folder,"MRS", "rda", str.split(target_protocol, '_')[0])
            os.makedirs(target_dir, exist_ok=True)

            # Construct new filename: subject_name_timepoint_<target_protocol>.rda
            new_filename = f"{subject_name}_{timepoint}_{target_protocol}.rda"
            target_path = os.path.join(target_dir, new_filename)

            try:
                shutil.move(rda_file, target_path)
                print(f"Moved {rda_file} to {target_path}")
            except Exception as e:
                print(f"Error moving {rda_file} to {target_path}: {e}")
        shutil.move(os.path.join(root_dir, subject_folder, "MRS"), os.path.join(subject_folder, session)) 

def clean_unknown_elements(data_dir: Path):
    """
    Removes (or logs) unknown files/folders in each subject folder.
    """
    for subj in data_dir.iterdir():
        if not subj.is_dir():
            continue
        for element in subj.iterdir():
            if not element.name.startswith('ses-'):
                logging.info(f"Unknown element in {subj.name}: {element.name}")
                # Uncomment the next line to remove unknown items:
                shutil.rmtree(element) if element.is_dir() else element.unlink()


def check_mrs_folder_structure(data_dir: Path, voi_list: list):
    """
    Verifies that for each subject the required files exist:
      - For each VOI: subject_VOI.rda under MRS/rda/VOI
      - T1w image in anat folder.
    """
    subjects = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    missing_items = []

    for subj in subjects:
        for session in subj.iterdir():
            session_name = session.name
            for voi in voi_list:
                rda_file = subj / session_name /'MRS' / 'rda' / voi / f"{subj.name}_{session_name[-3:]}_{voi}.rda"
                if not rda_file.exists():
                    missing_items.append(str(rda_file))
            subj_name_strip = subj.name.replace('_', '')
            anat_dir = subj / session_name /'anat'
            pattern = f"sub-{subj_name_strip}_{session_name}_run-*_T1w.nii.gz"
            t1_candidates = list(anat_dir.glob(pattern))
            if not t1_candidates[0].exists():
                missing_items.append(str(t1_candidates[0]))

    if missing_items:
        logging.warning("The following required files/folders are missing:")
        for item in missing_items:
            logging.warning(f" - {item}")
    else:
        logging.info("All required files and folders exist for all subjects.")


def read_rda_header(rda_filename: Path) -> dict:
    """
    Reads a Siemens MRS RDA file header and extracts metadata into a dictionary.
    """
    metadata = {}
    with rda_filename.open('r', errors='ignore') as file:
        lines = file.readlines()
    header_start = ">>> Begin of header <<<"
    header_end = ">>> End of header <<<"
    parsing = False
    for line in lines:
        line = line.strip()
        if line == header_start:
            parsing = True
            continue
        if line == header_end:
            break
        if parsing and ':' in line:
            key, value = line.split(':', 1)
            key, value = key.strip(), value.strip()
            try:
                metadata[key] = float(value)
            except ValueError:
                metadata[key] = value
    return metadata


def generate_voi_mask(nifti_volume, rda_file_path: Path) -> np.ndarray:
    """
    Generates a VOI segmentation mask for the given NIfTI volume based on the RDA metadata.
    """
    meta = read_rda_header(rda_file_path)
    half_fov_x = meta['FoVWidth'] / 2
    half_fov_y = meta['FoVHeight'] / 2
    half_fov_z = meta['FoV3D'] / 2

    row_vector = np.array([meta['RowVector[0]'], meta['RowVector[1]'], meta['RowVector[2]']]) * [-1, -1, 1]
    column_vector = np.array([meta['ColumnVector[0]'], meta['ColumnVector[1]'], meta['ColumnVector[2]']]) * [-1, -1, 1]
    cross_vector = np.cross(column_vector, row_vector)

    reference_pos = np.array([-meta['VOIPositionSag'], -meta['VOIPositionCor'], meta['VOIPositionTra']])
    voxel_indices = np.floor(np.linalg.inv(nifti_volume.affine) @ np.append(reference_pos, 1))
    _, s_vals, _ = np.linalg.svd(nifti_volume.affine[:3, :3])
    max_dims = np.floor(np.divide(
        np.array([meta['FoVWidth'], meta['FoVHeight'], meta['FoV3D']]),
        s_vals, where=s_vals != 0)
    ).flatten()

    mask = np.zeros(nifti_volume.shape, dtype=np.uint8)
    for x in range(int(voxel_indices[0] - max_dims[0]), int(voxel_indices[0] + max_dims[0])):
        for y in range(int(voxel_indices[1] - max_dims[1]), int(voxel_indices[1] + max_dims[1])):
            for z in range(int(voxel_indices[2] - max_dims[2]), int(voxel_indices[2] + max_dims[2])):
                world_coords = np.dot(nifti_volume.affine, np.array([x, y, z, 1]))
                disp = world_coords[:3] - reference_pos
                if (abs(np.dot(disp, row_vector)) < half_fov_x and
                    abs(np.dot(disp, column_vector)) < half_fov_y and
                    abs(np.dot(disp, cross_vector)) < half_fov_z):
                    mask[x, y, z] = 1
    return mask


def process_voxel_masks(data_dir: Path, voi_list: list):
    """
    For each subject and each VOI in voi_list, generates and saves the VOI mask.
    """
    for subj in data_dir.iterdir():
        for session in subj.iterdir():
            session_name = session.name
            if not session.is_dir():
                continue
            subject_id = subj.name
            subj_name_strip = subj.name.replace('_', '')
            anat_path = subj / session_name /'anat' 
            pattern = f"sub-{subj_name_strip}_{session_name}_run-*_T1w.nii.gz"
            t1_candidates = list(anat_path.glob(pattern))
            nifti_vol = nib.load(str(t1_candidates[0]))
            for voi in voi_list:
                rda_path = subj / session_name /'MRS' / 'rda' / voi / f"{subj.name}_{session_name[-3:]}_{voi}.rda"
                if rda_path.exists():
                    mask = generate_voi_mask(nifti_vol, rda_path)
                    mask_out = subj / session_name / 'MRS' / 'rda' / voi / f"MRS_mask_{voi}.nii"
                    nib.save(nib.Nifti1Image(mask, nifti_vol.affine), str(mask_out))
                    logging.info(f"Saved mask: {mask_out}")
                else:
                    logging.warning(f"Missing RDA file for {voi} in subject {subject_id}")


def reorient_nifti_files(data_dir: Path):
    """
    Reorients all NIfTI files in the dataset using FSL's Reorient2Std.
    """
    for subj in data_dir.iterdir():
        if not subj.is_dir():
            continue
        for nii_file in subj.rglob("*.nii*"):
            if "_reoriented" not in nii_file.name:
                logging.info(f"Reorienting: {nii_file.name}")
                reorient = fsl.Reorient2Std()
                reorient.inputs.in_file = str(nii_file)
                out_name =  str(nii_file.parent) + '/' + str.split(str(nii_file.name), '.')[0] + '_reoriented.nii.gz'
                reorient.inputs.out_file = str(nii_file.parent / out_name)
                reorient.inputs.output_type = 'NIFTI_GZ'
                reorient.run()
                logging.info(f"Saved reoriented file: {reorient.inputs.out_file}")


def run_spm_segmentation(t1_path: Path):
    """
    Runs SPM segmentation on a T1-weighted image.
    """
    job = spm.NewSegment()
    job.inputs.channel_files = str(t1_path)
    job.run()
    logging.info(f"SPM segmentation completed for {t1_path}")


def plot_qc_slices(t1_img, gm_img, voi_mask, output_path: Path):
    """
    Plots sagittal, axial, and coronal slices with overlays.
    """
    t1_data = t1_img.get_fdata()
    affine = t1_img.affine
    left_right = np.sign(affine[0, 0])
    left_label, right_label = ("R", "L") if left_right < 0 else ("L", "R")

    sagittal = np.linspace(t1_data.shape[0]*0.35, t1_data.shape[0]*0.65, 6, dtype=int)
    axial = np.linspace(t1_data.shape[2]*0.35, t1_data.shape[2]*0.65, 6, dtype=int)
    coronal = np.linspace(t1_data.shape[1]*0.35, t1_data.shape[1]*0.65, 6, dtype=int)

    fig, axes = plt.subplots(3, 6, figsize=(20, 10))
    for i, idx in enumerate(sagittal):
        axes[0, i].imshow(t1_data[idx, :, :].T, cmap='gray', origin='lower')
        axes[0, i].contour(gm_img[idx, :, :].T, levels=[0.5], colors='r', linewidths=0.5)
        axes[0, i].contour(voi_mask[idx, :, :].T, levels=[0.5], colors='b', linewidths=3)
        axes[0, i].set_title(f"Sagittal {idx}")
        axes[0, i].axis('off')
    for i, idx in enumerate(axial):
        axes[1, i].imshow(t1_data[:, :, idx], cmap='gray', origin='lower')
        axes[1, i].contour(gm_img[:, :, idx], levels=[0.5], colors='r', linewidths=0.5)
        axes[1, i].contour(voi_mask[:, :, idx], levels=[0.5], colors='b', linewidths=3)
        axes[1, i].set_title(f"Axial {idx}")
        axes[1, i].axis('off')
        axes[1, i].text(t1_data.shape[1]//2, 5, left_label, color='white', fontsize=12, weight='bold')
        axes[1, i].text(t1_data.shape[1]//2, t1_data.shape[0]*0.90, right_label, color='white', fontsize=12, weight='bold')
    for i, idx in enumerate(coronal):
        axes[2, i].imshow(t1_data[:, idx, :].T, cmap='gray', origin='lower')
        axes[2, i].contour(gm_img[:, idx, :].T, levels=[0.5], colors='r', linewidths=0.5)
        axes[2, i].contour(voi_mask[:, idx, :].T, levels=[0.5], colors='b', linewidths=3)
        axes[2, i].set_title(f"Coronal {idx}")
        axes[2, i].axis('off')
        axes[2, i].text(0.05, t1_data.shape[2]//2, left_label, color='white', fontsize=12, weight='bold')
        axes[2, i].text(t1_data.shape[0]*0.90, t1_data.shape[2]//2, right_label, color='white', fontsize=12, weight='bold')

    plt.savefig(str(output_path))
    plt.close()
    logging.info(f"QC image saved: {output_path}")


def filter_sd_columns(row, sd_threshold):
    """
    For a given DataFrame row, check all columns whose names contain 'SD'.
    If the value in such a column is not null or 'NA' and exceeds sd_threshold,
    mark that column and its immediate neighbors (previous and next columns) as 'NA'.
    
    Parameters:
        row (pd.Series): A row from the DataFrame.
        sd_threshold (float): The threshold above which values are considered outliers.
    
    Returns:
        pd.Series: The modified row.
    """
    for i, col in enumerate(row.index):
        if 'SD' in col:
            value = row[col]
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                continue
            if pd.notnull(value) and value != 'NA' and numeric_value > sd_threshold:
                if i > 0:
                    row.iloc[i - 1] = 'NA'
                row.iloc[i] = 'NA'
                if i < len(row.index) - 1:
                    row.iloc[i + 1] = 'NA'
    return row


def process_subjects(data_dir: Path, result_subdir: str, voi_list: list):
    """
    For each subject, performs SPM segmentation, calculates tissue fractions and water concentration,
    and generates QC images. The processing is repeated for each VOI.
    """
    results = []
    subjects = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name not in ['qc_images', 'tissue_fraction_wconc']])
    for subj in subjects:
        for session in subj.iterdir():
            session_name = session.name
            subj_id = subj.name
            subj_name_strip = subj.name.replace('_', '')
            anat_dir = subj / session_name /'anat'
            pattern = f"sub-{subj_name_strip}_{session_name}_run-*_T1w_reoriented.nii.gz"
            t1_candidates = list(anat_dir.glob(pattern))
            nifti_vol = nib.load(str(t1_candidates[0]))            
            t1_nii =  anat_dir / str(t1_candidates[0])[:-3]
            if not t1_nii.exists():
                nib.save(nifti_vol, t1_nii)
            logging.info(f"Processing subject: {subj_id}")
            subj_name_strip = subj.name.replace('_', '')
            img = nib.load(str(t1_nii))
            if not list(anat_dir.glob("c*")):
                run_spm_segmentation(t1_nii)
            seg_patterns = {
                'gm': f'c1sub-{subj_name_strip}_{session_name}_run*_T1w_reoriented.nii',
                'wm': f'c2sub-{subj_name_strip}_{session_name}_run*_T1w_reoriented.nii',
                'csf': f'c3sub-{subj_name_strip}_{session_name}_run*_T1w_reoriented.nii'
            }
            seg_files = {}
            for label, pattern in seg_patterns.items():
                # Glob for files matching the pattern:
                matches = list(anat_dir.glob(pattern))
                if matches:
                    # If multiple matches exist, pick the first or handle otherwise
                    seg_files[label] = matches[0]
                else:
                    seg_files[label] = None
            segmented_imgs = {k: nib.load(str(v)) for k, v in seg_files.items() if v.exists()}
            segmented_data = {k: img.get_fdata() for k, img in segmented_imgs.items()}
    
            for voi in voi_list:
                rda_file = subj / session_name / 'MRS' / 'rda' / voi / f'{subj.name}_{session_name[-3:]}_{voi}.rda'
                mask_path = subj / session_name / 'MRS' / 'rda' / voi / f'MRS_mask_{voi}_reoriented.nii.gz'
                if not (rda_file.exists() and mask_path.exists()):
                    continue
                mask_img = nib.load(str(mask_path))
                mask_data = mask_img.get_fdata()
                pgm = np.sum(segmented_data.get('gm', 0) * mask_data)
                pwm = np.sum(segmented_data.get('wm', 0) * mask_data)
                pcsf = np.sum(segmented_data.get('csf', 0) * mask_data)
                total = pgm + pwm + pcsf
                if total > 0:
                    pgm, pwm, pcsf = (100 * pgm / total, 100 * pwm / total, 100 * pcsf / total)
                else:
                    pgm = pwm = pcsf = 0
                wcon = ((43300 * pgm/100) + (35880 * pwm/100) + (5556 * pcsf/100)) / (1 - pcsf/100) if pcsf < 100 else 0
                results.append([f'{subj.name}_{session_name[-3:]}_{voi}', pgm, pwm, pcsf, wcon])
                # Generate QC image
                qc_dir = data_dir / 'qc_images'
                qc_dir.mkdir(exist_ok=True)
                qc_output = qc_dir / f'{subj.name}_{voi}_qc.png'
                plot_qc_slices(img, segmented_data.get('gm', np.zeros_like(img.get_fdata())), mask_data, qc_output)
    result_dir = data_dir / result_subdir
    result_dir.mkdir(exist_ok=True)
    result_file = result_dir / 'tissue_fraction_wconc.xlsx'
    df = pd.DataFrame(results, columns=['ID', 'GM', 'WM', 'CSF', 'wcon'])
    # Apply the SD filtering to each row of the DataFrame using the helper function.
    SD_THRESHOLD = 15
    df = df.apply(lambda r: filter_sd_columns(r, SD_THRESHOLD), axis=1)
    df.to_excel(result_file, index=False)
    logging.info(f"Results saved to {result_file}")


def format_control_file(subject: str, results_path: Path, voi: str, session_name: str, control_file: Path, wconc: str, echot: str, deltat: str):
    """
    Creates an LCModel control file with the given parameters.
    """
    lines = [
        " $LCMODL",
        " key = 210387309",
        f" srcraw = '{results_path}/{subject}_{session_name[-3:]}_{voi}.rda'",
        f" srch2o = '{results_path}/{subject}_{session_name[-3:]}_{voi}_H2O.rda'",
        f" savdir ='{results_path}'",
        f" subbas = T\n ppmst = 4.0\n ppmend = 0.2\n nunfil = 2048\n ltable = 7\n lps = 8\n lcsv = 11\n lcoord = 9\n hzpppm = 1.2325e+02\n dows=T\n doecc = T\n deltat = {deltat}",
        f" filtab = '{results_path}/{subject}_{session_name[-3:]}_{voi}.table '",
        f" filraw = '{results_path}/{subject}_{session_name[-3:]}_{voi}.RAW'",
        f" filps = '{results_path}/{subject}_{session_name[-3:]}_{voi}_ps'",
        f" filh2o = '{results_path}/{subject}_{session_name[-3:]}_{voi}.H2O'",
        f" filcsv = '{results_path}/{subject}_{session_name[-3:]}_{voi}.csv'",
        f" filcoo = '{results_path}/coord'",
        f" filbas ='{LCMODEL_PARAMS['FILBAS']}'",
        f" wconc = {wconc}",
        f" echot = {echot}",
        " $END"
    ]
    with control_file.open('w') as f:
        f.write("\n".join(lines))


def spectral_fitting(mrs_general: Path, wconc_db: pd.DataFrame, voi_list: list):
    """
    Performs spectral fitting for each subject and each VOI.
    """
    subjects = sorted([d for d in mrs_general.iterdir() if d.is_dir() and d.name not in ['qc_images', 'tissue_fraction_wconc']])
    for subj in subjects:
        for session in subj.iterdir():
            session_name = session.name
            subject_id = subj.name
            logging.info(f"Spectral fitting for subject {subject_id}")
            for voi in voi_list:
                try:
                    wconc_val = str(wconc_db.loc[f"{subject_id}_{session_name[-3:]}_{voi}"].iloc[3])
                except Exception as e:
                    logging.warning(f"No wconc for {subject_id}_{voi}: {e}")
                    continue
    
                
                results_path = session /"MRS" / "results"
                results_path.mkdir(exist_ok=True)
                try:
                    # Write raw files using suspect
                    rda_file = session / "MRS" / "rda" / voi / f"{subject_id}_{session_name[-3:]}_{voi}.rda"
                    rda_file_h2o = session / "MRS" / "rda" / voi / f"{subject_id}_{session_name[-3:]}_{voi}_H2O.rda"
                    # Write LCModel input files using suspect (this is a placeholder)
                    suspect.io.lcmodel.write_all_files(str(results_path / f"{subject_id}_{session_name[-3:]}_{voi}.RAW"),
                                                       suspect.io.load_rda(str(rda_file)),
                                                       suspect.io.load_rda(str(rda_file_h2o)),
                                                       params=LCMODEL_PARAMS)
                    # Determine which control file parameters to use based on file content.
                    with rda_file.open('r', errors='ignore') as f:
                        for line in f:
                            if line.startswith('ModelName: TrioTim'):
                                control_file = results_path / f"{subject_id}_{session_name[-3:]}_{voi}_sl0.CONTROL"
                                format_control_file(subject_id, results_path, voi, session_name, control_file, wconc_val, echot, deltat="2.000e-04")
                                break
                            elif line.startswith('ModelName: Prisma_fit'):
                                control_file = results_path / f"{subject_id}_{session_name[-3:]}_{voi}_sl0.CONTROL"
                                format_control_file(subject_id, results_path, voi, session_name, control_file, wconc_val, echot, deltat="4.000e-04")
                                break
                    lcmodel_command = f"$HOME/.lcmodel/bin/lcmodel < {results_path}/{subject_id}_{session_name[-3:]}_{voi}_sl0.CONTROL"
                    os.system(lcmodel_command)
                except Exception as e:
                    logging.warning(f"Spectral fitting failed for {subject_id}_{voi}: {e}")


def qc_processing(mrs_general: Path, wconc_db: pd.DataFrame, voi_list: list):
    """
    Processes QC metrics and writes out a summary CSV.
    """
    SD_THRESHOLD = 15
    qc_output = pd.DataFrame()
    subjects = sorted([d for d in mrs_general.iterdir() if d.is_dir() and d.name not in ['qc_images', 'tissue_fraction_wconc']])
    
    for subj in subjects:
        subject_id = subj.name
        for session in subj.iterdir():
            session_name = session.name
            subject_id = subj.name
            for voi in voi_list:
                try:
                    csv_path = session / "MRS" / "results" / f"{subject_id}_{session_name[-3:]}_{voi}.csv"
                    df_csv = pd.read_csv(csv_path).drop(['Row', ' Col'], axis=1)
                    df_csv.insert(0, 'ID', subject_id)
                    df_csv.set_index('ID', inplace=True)
                    df_csv.columns = df_csv.columns.str.strip().str.replace(" ", "_")
                    # Extract FWHM/SNR from table file
                    FWHM, SNR = 'NA', 'NA'
                    table_path = session / "MRS" / "results" / f"{subject_id}_{session_name[-3:]}_{voi}.table"
                    with table_path.open() as f:
                        for line in f:
                            if line.startswith('  FWHM'):
                                parts = line.split()
                                FWHM = float(parts[parts.index('FWHM') + 2])
                                SNR = float(parts[parts.index('S/N') + 2])
                    df_csv.insert(0, 'FWHM', FWHM)
                    df_csv.insert(1, 'SNR', SNR)
                    qc_val = '1' if (FWHM != 'NA' and SNR != 'NA' and FWHM <= 0.1 and SNR > 10) else '0'
                    df_csv.insert(0, 'QC', qc_val)
                    # Add VOI suffix to columns
                    df_csv = df_csv.add_suffix(f"_{voi}")
                except Exception as e:
                    logging.error(f"Error processing QC for {subject_id}_{voi}: {e}")
                    df_csv = pd.DataFrame({f"QC_{voi}": ['NA'], f"FWHM_{voi}": ['NA'], f"SNR_{voi}": ['NA']}, index=[subject_id])
                    # Merge results for each VOI
                if subject_id in qc_output.index:
                    for col in df_csv.columns:
                        qc_output.loc[subject_id, col] = df_csv.loc[subject_id, col]
                else:
                    qc_output = pd.concat([qc_output, df_csv], axis=0)
            
            # Append additional subject-level info from RDA header if available
            try:
                rda_file = next((session / "MRS" / "rda" / voi / f"{subject_id}_{session_name[-3:]}_{voi}.rda" for voi in voi_list if (session / "MRS" / "rda" / voi / f"{subject_id}_{session_name[-3:]}_{voi}.rda").exists()), None)
                if rda_file:
                    with rda_file.open('r', errors='ignore') as f:
                        for line in f:
                            if line.startswith('StudyDate:'):
                                study_date = line.split()[1].strip()
                                formatted_date = datetime.strptime(study_date, '%Y%m%d').strftime('%d/%m/%Y')
                                qc_output.at[subject_id, 'mrs_acquisition_date'] = formatted_date
                            if line.startswith('PatientSex:'):
                                qc_output.at[subject_id, 'sex'] = line.split()[1].strip()
                            if line.startswith('PatientBirthDate:'):
                                try:
                                    birthdate = line.split()[1].strip()
                                    formatted_bd = datetime.strptime(birthdate, '%Y%m%d').strftime('%d/%m/%Y')
                                    qc_output.at[subject_id, 'birthdate'] = formatted_bd
                                except Exception:
                                    logging.warning(f"{subject_id} has anonymized birthdate")
            except Exception as e:
                logging.warning(f"Error reading additional info for {subject_id}: {e}")
    
    # Append wconc information from wconc_db
    for subj in subjects:
        subject_id = subj.name
        for voi in voi_list:
            try:
                wconc_val = str(wconc_db.loc[f"{subject_id}_{session_name[-3:]}_{voi}"][3])
                qc_output.at[subject_id, f'wconc_{voi}'] = wconc_val
                gm_val = str(wconc_db.loc[f"{subject_id}_{session_name[-3:]}_{voi}"][0])
                qc_output.at[subject_id, f'GM_{voi}'] = gm_val
                wm_val = str(wconc_db.loc[f"{subject_id}_{session_name[-3:]}_{voi}"][1])
                qc_output.at[subject_id, f'WM_{voi}'] = wm_val
                csf_val = str(wconc_db.loc[f"{subject_id}_{session_name[-3:]}_{voi}"][2])
                qc_output.at[subject_id, f'CSF_{voi}'] = csf_val
            except Exception:
                logging.warning(f"No wconc in db for {subject_id}_{session_name[-3:]}_{voi}")
    
    # Apply SD filtering on the entire QC DataFrame
    qc_output = qc_output.apply(lambda r: filter_sd_columns(r, SD_THRESHOLD), axis=1)
    
    output_csv = mrs_general / 'test_retest.csv'
    qc_output.to_csv(str(output_csv))
    logging.info(f"QC summary saved to {output_csv}")


# ----------------------------
# Main Processing Pipeline
# ----------------------------




def main():
    # Step 1: Organize folder structure
    # first generate a copy of the raw input
    shutil.copytree(str(MRS_DATA_DIR), str(MRS_DATA_DIR) + '_raw')    
    for subject_name in os.listdir(MRS_DATA_DIR):
        process_mrs_directory(MRS_DATA_DIR, protocol_dict, subject_name, '001', heuristic_path)
    clean_unknown_elements(MRS_DATA_DIR)
    check_mrs_folder_structure(MRS_DATA_DIR, VOI_LIST)

    # Step 2: Generate VOI masks
    process_voxel_masks(MRS_DATA_DIR, VOI_LIST)

    # Step 3: Reorient all NIfTI files
    reorient_nifti_files(MRS_DATA_DIR)

    # Step 4: Process subjects for tissue fractions & water concentration
    process_subjects(MRS_DATA_DIR, result_subdir='tissue_fraction_wconc', voi_list=VOI_LIST)

    # Step 5: Spectral fitting (requires a wconc DB Excel file)
    wconc_db_path = MRS_DATA_DIR / 'tissue_fraction_wconc' / 'tissue_fraction_wconc.xlsx'
    if wconc_db_path.exists():
        wconc_db = pd.read_excel(str(wconc_db_path), index_col=0)
        spectral_fitting(MRS_DATA_DIR, wconc_db, VOI_LIST)
    else:
        logging.warning(f"wconc DB file not found at {wconc_db_path}")

    # Step 6: QC processing
    if wconc_db_path.exists():
        wconc_db = pd.read_excel(str(wconc_db_path), index_col=0)
        qc_processing(MRS_DATA_DIR, wconc_db, VOI_LIST)
    else:
        logging.warning(f"wconc DB file not found at {wconc_db_path}")


if __name__ == '__main__':
    main()
