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

# Base directories (change these as needed)
MRS_GENERAL_DIR = Path('/media/maria/5016523d-1bee-4995-9853-7e1dcc60bd0f/home/piedra_filosofal_backup/paper_mrs_resting_nmda_2025/processed_mrs')
MRS_DATA_DIR = MRS_GENERAL_DIR / 'MRS_data_organized_processed'

# Define the list of base VOIs; to add more VOIs, simply update this list.
VOI_LIST = ['dmPF', 'mTL']

# LCModel parameters (if needed by spectral fitting)
LCMODEL_PARAMS = {
    "FILBAS": "/home/maria/.lcmodel/basis-sets/3t/press_te30_3t_v3.basis",
    "LCSV": True,
    "LTABLE": True
}

# ----------------------------
# Helper Functions
# ----------------------------

def reorganize_folders(data_dir: Path):
    """
    Ensures each subject folder has the expected subfolders:
      - MRS (with subfolder 'rda')
      - anat
    Moves files accordingly.
    """
    for subj in data_dir.iterdir():
        if not subj.is_dir():
            continue

        # Create MRS folder if missing and move any 'rda' folder into it.
        mrs_dir = subj / 'MRS'
        if not mrs_dir.exists():
            logging.info(f"Creating MRS folder for subject {subj.name}")
            mrs_dir.mkdir()
            rda_src = subj / 'rda'
            if rda_src.exists():
                shutil.move(str(rda_src), str(mrs_dir / 'rda'))
            else:
                logging.warning(f"Subject {subj.name} does not have a 'rda' folder.")

        # Create anat folder and move T1w file if needed.
        anat_dir = subj / 'anat'
        if not anat_dir.exists():
            logging.info(f"Creating anat folder for subject {subj.name}")
            anat_dir.mkdir()
            t1_file = subj / f"T1w_{subj.name}.nii.gz"
            if t1_file.exists():
                shutil.move(str(t1_file), str(anat_dir))
            else:
                logging.warning(f"T1w file missing for subject {subj.name}.")


def clean_unknown_elements(data_dir: Path):
    """
    Removes (or logs) unknown files/folders in each subject folder.
    """
    for subj in data_dir.iterdir():
        if not subj.is_dir():
            continue
        for element in subj.iterdir():
            if element.name not in ['MRS', 'anat']:
                logging.info(f"Unknown element in {subj.name}: {element.name}")
                # Uncomment the next line to remove unknown items:
                # shutil.rmtree(element) if element.is_dir() else element.unlink()


def clean_anat_folder(data_dir: Path):
    """
    In each anat folder, remove any file that is not the expected T1w files.
    Also rename T1w files to follow the convention: subject_T1w.nii.gz.
    """
    for subj in data_dir.iterdir():
        anat_dir = subj / 'anat'
        if not anat_dir.exists():
            continue
        for file in anat_dir.iterdir():
            expected_names = {f"{subj.name}_T1w.json", f"{subj.name}_T1w.nii.gz"}
            if file.name not in expected_names:
                logging.info(f"Removing unexpected file {file.name} in {subj.name}/anat")
                file.unlink()

        # Rename files if needed
        for file in anat_dir.iterdir():
            if file.name.startswith("T1w_") and not file.name.startswith(f"{subj.name}_"):
                new_name = f"{subj.name}_T1w.nii.gz"
                logging.info(f"Renaming {file.name} to {new_name} in {subj.name}/anat")
                file.rename(anat_dir / new_name)


def determine_sequence_name(rda_file: Path) -> str:
    """
    Reads an .rda file and returns a sequence name based on header content.
    This implementation uses keyword matching. Extend this function
    if you add more VOIs or need different logic.
    """
    sequence_name = None
    try:
        with rda_file.open('r', errors='ignore') as f:
            lines = f.readlines()
        for line in lines:
            if 'PatientName:' in line:
                if 'pfcglu' in line:
                    sequence_name = 'dmPF'
                    break
                elif 'pfch2o' in line:
                    sequence_name = 'dmPF_H2O'
                    break
                elif 'hcglu' in line:
                    sequence_name = 'mTL'
                    break
                elif 'hch2o' in line:
                    sequence_name = 'mTL_H2O'
                    break
            elif 'SeriesDescription:' in line:
                # Fallback: extract series description and use as sequence name
                sequence_name = line.split('SeriesDescription:')[-1].strip()
                break
    except Exception as e:
        logging.error(f"Error reading {rda_file}: {e}")

    if sequence_name is None:
        logging.warning(f"Could not determine sequence name for {rda_file}")
        sequence_name = "unknown"
    return sequence_name


def rename_rda_files(data_dir: Path, voi_list: list):
    """
    Walks through the data folder, reads each .rda file header, determines the VOI,
    and renames the file following the convention: subject_VOI.rda.
    Files with water-unsuppressed acquisitions should include an _H2O suffix.
    The file is then placed into a folder named after the base VOI (e.g. "dmPF" or "mTL").
    """
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.rda'):
                file_path = Path(root) / file
                seq_name = determine_sequence_name(file_path)
                # Remove any "_H2O" suffix for folder assignment
                base_voi = seq_name.replace("_H2O", "")
                if base_voi not in voi_list:
                    logging.warning(f"Sequence {seq_name} in {file_path} does not match any known VOI; skipping.")
                    continue
                # Assume subject ID is 4 folders up in the structure
                subject_id = Path(root).parts[-4]
                new_filename = f"{subject_id}_{seq_name}.rda"
                new_path = Path(root) / new_filename
                logging.info(f"Renaming {file_path.name} to {new_filename}")
                file_path.rename(new_path)


def create_voi_folders(data_dir: Path, voi_list: list):
    """
    For each subject, creates subfolders under MRS/rda for each VOI in voi_list.
    """
    for subj in data_dir.iterdir():
        if not subj.is_dir():
            continue
        rda_dir = subj / 'MRS' / 'rda'
        if not rda_dir.exists():
            continue
        for voi in voi_list:
            voi_dir = rda_dir / voi
            if not voi_dir.exists():
                logging.info(f"Creating folder {voi_dir}")
                voi_dir.mkdir()


def move_rda_files_to_voi_folders(data_dir: Path, voi_list: list):
    """
    Moves .rda files into their corresponding VOI folders.
    Files are assigned based on whether the filename contains the VOI name.
    """
    for subj in data_dir.iterdir():
        if not subj.is_dir():
            continue
        rda_dir = subj / 'MRS' / 'rda'
        if not rda_dir.exists():
            continue
        for file in rda_dir.glob("*.rda"):
            for voi in voi_list:
                if voi in file.name:
                    target_dir = rda_dir / voi
                    logging.info(f"Moving {file.name} to {target_dir}")
                    shutil.move(str(file), str(target_dir / file.name))
                    break


def remove_empty_voi_folders(data_dir: Path, voi_list: list):
    """
    Removes any VOI folders that are empty.
    """
    for subj in data_dir.iterdir():
        if not subj.is_dir():
            continue
        for voi in voi_list:
            voi_dir = subj / 'MRS' / 'rda' / voi
            if voi_dir.exists() and not any(voi_dir.iterdir()):
                logging.info(f"Removing empty folder {voi_dir}")
                shutil.rmtree(voi_dir)


def check_mrs_folder_structure(mrs_general: Path, data_dir: Path, voi_list: list):
    """
    Verifies that for each subject the required files exist:
      - For each VOI: subject_VOI.rda under MRS/rda/VOI
      - T1w image in anat folder.
    """
    subjects = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
    missing_items = []

    for subj in subjects:
        subject_name = subj.name.replace('sub-', '')
        for voi in voi_list:
            rda_file = subj / 'MRS' / 'rda' / voi / f"{subject_name}_{voi}.rda"
            if not rda_file.exists():
                missing_items.append(str(rda_file))
        t1_file = subj / 'anat' / f"{subj.name}_T1w.nii.gz"
        if not t1_file.exists():
            missing_items.append(str(t1_file))

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
        if not subj.is_dir():
            continue
        subject_id = subj.name
        anat_path = subj / 'anat' / f"{subject_id}_T1w.nii.gz"
        if not anat_path.exists():
            anat_path = anat_path.with_suffix('')  # try without .gz
        nifti_vol = nib.load(str(anat_path))
        for voi in voi_list:
            rda_path = subj / 'MRS' / 'rda' / voi / f"{subject_id}_{voi}.rda"
            if rda_path.exists():
                mask = generate_voi_mask(nifti_vol, rda_path)
                mask_out = subj / 'MRS' / 'rda' / voi / f"MRS_mask_{voi}.nii"
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
                out_name = nii_file.stem + '_reoriented.nii.gz'
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
        subj_id = subj.name
        anat_dir = subj / 'anat'
        t1_nii_gz = anat_dir / f'{subj_id}_T1w_reoriented.nii.gz'
        t1_nii = anat_dir / f'{subj_id}_T1w_reoriented.nii'
        if not t1_nii.exists():
            img = nib.load(str(t1_nii_gz))
            nib.save(img, str(t1_nii))
        logging.info(f"Processing subject: {subj_id}")
        img = nib.load(str(t1_nii))
        # Run segmentation if not done
        seg_files = {
            'gm': anat_dir / f'c1{subj_id}_T1w_reoriented.nii',
            'wm': anat_dir / f'c2{subj_id}_T1w_reoriented.nii',
            'csf': anat_dir / f'c3{subj_id}_T1w_reoriented.nii'
        }
        if not seg_files['gm'].exists():
            run_spm_segmentation(t1_nii)
        segmented_imgs = {k: nib.load(str(v)) for k, v in seg_files.items() if v.exists()}
        segmented_data = {k: img.get_fdata() for k, img in segmented_imgs.items()}

        for voi in voi_list:
            rda_file = subj / 'MRS' / 'rda' / voi / f'{subj_id}_{voi}.rda'
            mask_path = subj / 'MRS' / 'rda' / voi / f'MRS_mask_{voi}_reoriented.nii.gz'
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
            results.append([f'{subj_id}_{voi}', pgm, pwm, pcsf, wcon])
            # Generate QC image
            qc_dir = data_dir / 'qc_images'
            qc_dir.mkdir(exist_ok=True)
            qc_output = qc_dir / f'{subj_id}_{voi}_qc.png'
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


def format_control_file(subject: str, results_path: Path, voi: str, control_file: Path, wconc: str, echot: str, deltat: str):
    """
    Creates an LCModel control file with the given parameters.
    """
    lines = [
        " $LCMODL",
        " key = 210387309",
        f" srcraw = '{results_path}/{subject}_{voi}.rda'",
        f" srch2o = '{results_path}/{subject}_{voi}_H2O.rda'",
        f" savdir ='{results_path}'",
        f" subbas = T\n ppmst = 4.0\n ppmend = 0.2\n nunfil = 2048\n ltable = 7\n lps = 8\n lcsv = 11\n lcoord = 9\n hzpppm = 1.2325e+02\n dows=T\n doecc = T\n deltat = {deltat}",
        f" filtab = '{results_path}/{subject}_{voi}.table '",
        f" filraw = '{results_path}/{subject}_{voi}.RAW'",
        f" filps = '{results_path}/{subject}_{voi}_ps'",
        f" filh2o = '{results_path}/{subject}_{voi}.H2O'",
        f" filcsv = '{results_path}/{subject}_{voi}.csv'",
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
        subject_id = subj.name
        logging.info(f"Spectral fitting for subject {subject_id}")
        for voi in voi_list:
            try:
                wconc_val = str(wconc_db.loc[f"{subject_id}_{voi}"][3])
            except Exception as e:
                logging.warning(f"No wconc for {subject_id}_{voi}: {e}")
                continue

            echot = "30"
            results_path = subj / "MRS" / "results"
            results_path.mkdir(exist_ok=True)
            try:
                # Write raw files using suspect
                rda_file = subj / "MRS" / "rda" / voi / f"{subject_id}_{voi}.rda"
                rda_file_h2o = subj / "MRS" / "rda" / voi / f"{subject_id}_{voi}_H2O.rda"
                # Write LCModel input files using suspect (this is a placeholder)
                suspect.io.lcmodel.write_all_files(str(results_path / f"{subject_id}_{voi}.RAW"),
                                                   suspect.io.load_rda(str(rda_file)),
                                                   suspect.io.load_rda(str(rda_file_h2o)),
                                                   params=LCMODEL_PARAMS)
                # Determine which control file parameters to use based on file content.
                with rda_file.open('r', errors='ignore') as f:
                    for line in f:
                        if line.startswith('ModelName: TrioTim'):
                            control_file = results_path / f"{subject_id}_{voi}_sl0.CONTROL"
                            format_control_file(subject_id, results_path, voi, control_file, wconc_val, echot, deltat="2.000e-04")
                            break
                        elif line.startswith('ModelName: Prisma_fit'):
                            control_file = results_path / f"{subject_id}_{voi}_sl0.CONTROL"
                            format_control_file(subject_id, results_path, voi, control_file, wconc_val, echot, deltat="4.000e-04")
                            break
                lcmodel_command = f"$HOME/.lcmodel/bin/lcmodel < {results_path}/{subject_id}_{voi}_sl0.CONTROL"
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
        for voi in voi_list:
            try:
                csv_path = mrs_general / subject_id / "MRS" / "results" / f"{subject_id}_{voi}.csv"
                df_csv = pd.read_csv(csv_path).drop(['Row', ' Col'], axis=1)
                df_csv.insert(0, 'ID', subject_id)
                df_csv.set_index('ID', inplace=True)
                df_csv.columns = df_csv.columns.str.strip().str.replace(" ", "_")
                # Extract FWHM/SNR from table file
                FWHM, SNR = 'NA', 'NA'
                table_path = mrs_general / subject_id / "MRS" / "results" / f"{subject_id}_{voi}.table"
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
                df_csv = df_csv.add_suffix(f"_{voi.upper()}")
            except Exception as e:
                logging.error(f"Error processing QC for {subject_id}_{voi}: {e}")
                df_csv = pd.DataFrame({f"QC_{voi.upper()}": ['NA'], f"FWHM_{voi.upper()}": ['NA'], f"SNR_{voi.upper()}": ['NA']}, index=[subject_id])
            # Merge results for each VOI
            if subject_id in qc_output.index:
                qc_output = qc_output.join(df_csv, how='outer')
            else:
                qc_output = pd.concat([qc_output, df_csv])
        
        # Append additional subject-level info from RDA header if available
        try:
            rda_file = next((mrs_general / subject_id / "MRS" / "rda" / voi / f"{subject_id}_{voi}.rda" for voi in voi_list if (mrs_general / subject_id / "MRS" / "rda" / voi / f"{subject_id}_{voi}.rda").exists()), None)
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
                wconc_val = str(wconc_db.loc[f"{subject_id}_{voi}"][3])
                qc_output.at[subject_id, f'wconc_{voi}'] = wconc_val
            except Exception:
                logging.warning(f"No wconc in db for {subject_id}_{voi}")
    
    # Apply SD filtering on the entire QC DataFrame
    qc_output = qc_output.apply(lambda r: filter_sd_columns(r, SD_THRESHOLD), axis=1)
    
    output_csv = mrs_general / 'new_nmda.csv'
    qc_output.to_csv(str(output_csv))
    logging.info(f"QC summary saved to {output_csv}")


# ----------------------------
# Main Processing Pipeline
# ----------------------------

def main():
    # Step 1: Organize folder structure
    reorganize_folders(MRS_DATA_DIR)
    clean_unknown_elements(MRS_DATA_DIR)
    clean_anat_folder(MRS_DATA_DIR)

    # Step 2: Rename and sort RDA files
    rename_rda_files(MRS_DATA_DIR, VOI_LIST)
    create_voi_folders(MRS_DATA_DIR, VOI_LIST)
    move_rda_files_to_voi_folders(MRS_DATA_DIR, VOI_LIST)
    remove_empty_voi_folders(MRS_DATA_DIR, VOI_LIST)
    check_mrs_folder_structure(MRS_GENERAL_DIR, MRS_DATA_DIR, VOI_LIST)

    # Step 3: Generate VOI masks
    process_voxel_masks(MRS_DATA_DIR, VOI_LIST)

    # Step 4: Reorient all NIfTI files
    reorient_nifti_files(MRS_DATA_DIR)

    # Step 5: Process subjects for tissue fractions & water concentration
    process_subjects(MRS_DATA_DIR, result_subdir='tissue_fraction_wconc', voi_list=VOI_LIST)

    # Step 6: Spectral fitting (requires a wconc DB Excel file)
    wconc_db_path = MRS_GENERAL_DIR / 'tissue_fraction_wconc' / 'tissue_fraction_wconc.xlsx'
    if wconc_db_path.exists():
        wconc_db = pd.read_excel(str(wconc_db_path), index_col=0)
        spectral_fitting(MRS_GENERAL_DIR, wconc_db, VOI_LIST)
    else:
        logging.warning(f"wconc DB file not found at {wconc_db_path}")

    # Step 7: QC processing
    if wconc_db_path.exists():
        wconc_db = pd.read_excel(str(wconc_db_path), index_col=0)
        qc_processing(MRS_GENERAL_DIR, wconc_db, VOI_LIST)
    else:
        logging.warning(f"wconc DB file not found at {wconc_db_path}")


if __name__ == '__main__':
    main()
