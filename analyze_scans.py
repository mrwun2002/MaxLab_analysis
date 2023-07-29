
import os
import glob


#First, find the scans in the filetree.

#The filestream goes 'project_name/date/chip_num/scan_type/assay_number
#The resulting folder has a file 'data.raw.h5' that has the raw data
#as well as an 'Analysis' folder.

parent_folder = 'E:'
project_name = 'Summer_2023_Batch_2'


