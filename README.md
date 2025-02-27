# Single-Voxel MRS Processing Pipeline

This project provides a Python-based processing pipeline for single-voxel PRESS Magnetic Resonance Spectroscopy (MRS) data. It utilizes:

SPM to calculate tissue fractions and water concentration for partial volume correction.
LCModel for spectral fitting and metabolite quantification.
This is a refactored version created with the help of ChatGPT, and I have not tested this refactored version yet.

🚀 Feedback is highly welcome! If you test it and find any issues or have suggestions for improvement, please let me know.

## Features

- **Folder Organization:**  
  Creates necessary subfolders (e.g., `MRS`, `anat`) for each subject and moves files into the correct locations.

- **File Renaming & Sorting:**  
  Reads RDA headers to determine the VOI and renames files following the `subject_VOI.rda` convention. Supports multiple VOIs via an easily configurable list.

- **VOI Mask Generation:**  
  Uses metadata from RDA files to generate a segmentation mask for each VOI.

- **Image Reorientation:**  
  Reorients all NIfTI images using FSL’s `Reorient2Std` to ensure a standard orientation.

- **Tissue Segmentation:**  
  Runs SPM segmentation on T1-weighted images to calculate tissue fractions and water concentration.

- **Spectral Fitting:**  
  Generates LCModel input files for spectral fitting.

- **QC Processing & SD Filtering:**  
  Reads QC metrics from CSV and table files, then applies a custom filter (`filter_sd_columns`) to flag potential outliers based on standard deviation (SD) values.

## Dependencies

- [Nibabel](https://nipy.org/nibabel/)
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Suspect](https://github.com/suspect-toolbox) (for RDA and LCModel file handling)
- [Nipype](https://nipype.readthedocs.io/) (with FSL and SPM interfaces)
- [FSL](https://fsl.fmrib.ox.ac.uk/fsl/oldwiki/Fslutils.html?highlight=%28%5CbCategoryOther%5Cb%29) (for image reorientation)
- MATLAB with SPM installed (for segmentation)
- [LCModel](http://www.lcmodel.com/) (for spectral fitting)

Ensure that all paths (e.g., to MATLAB, basis sets, and data directories) are correctly set in the configuration section of the code.

## Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/mrs-processor.git
   cd mrs-processor
