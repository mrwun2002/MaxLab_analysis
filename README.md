# MaxLab_analysis
Analysis files from the MaxLab MaxOne HD-MEA system.

maxlab_analysis.py includes general utils for data analysis, including functions to load data from files output by the MaxOne using the Assay class.

assay.py includes the Assay class, a class that stores info about individual assays. Also includes functions to aid in the analysis of assays.

analysis_pipeline.py includes an example of how to run versatile analyses.

All jupyter notebooks are analysis scripts.


Instructions:
Copy "project" folder produced by MaxLab system into the hard drive. Build all raw npy and spike array files by calling mla.load_assays_from_project(). After all files are built, analysis can proceed, using the "project" folder as the parent folder.
See analysis_pipeline.py or pca_all_chips.ipynb for examples on how to use assay.load_build_npy() to save and load different analyses.

