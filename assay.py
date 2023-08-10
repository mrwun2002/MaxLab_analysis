
from pathlib import Path
from datetime import datetime
import maxlab_analysis as mla
import numpy as np
import pandas as pd
import sys


class Assay:
    '''
    A container for all information about a single assay + some processing.
    path: Path
        The path to the folder that the assay (and all processing of the assay) is in. 
    assay_number: int
        the unique assay number (per project) assigned by the MaxLab system.
    assay_type: str
        the assay type. For now, I've only been working with NetworkAssays.
    chip_number: str
        the chip number
    date: datetime.date
        the date that the assay was conducted
    project: str
        the name of the project
    raw_data_path: Path
        the full path to the raw data
        
    '''
    def __init__(self, path, build_raw_npy = True, build_spike_array = True, build_umap = True):
        '''
        Creates an Assay object from a path to a folder. Takes in a filepath to the folder with all data files in the form of a Path object or a string.
        the 'build' parameters tell you whether to create a file if it does not yet exist. They are either True or False.
        '''
        self.path = Path(path)
        self.assay_number = None
        self.assay_type = None
        self.chip_number = None
        self.date = None
        self.project = None
        
        self.raw_data_path = Path(self.path, 'data.raw.h5', mmap_mode = 'r')
        self.spike_df = mla.load_spikes_from_file(Path(self.raw_data_path), voltage_threshold=-10)
        self.spike_array = None
        self.raw_npy = None

        #make spike array
        try:
            self.spike_array = np.load(Path(self.path, 'spike_array.npy'), mmap_mode = 'r')
        except FileNotFoundError:
            if build_spike_array:
                print('building spike array in ' + str(self.path))
                #replaces spike_df with a version that also includes bin_id of each spike.
                self.spike_df = mla.bin_spike_data(self.spike_df, save = True, save_name = Path(self.path, 'spike_array'))[1]
                self.spike_array = np.load(Path(self.path, 'spike_array.npy'), mmap_mode = 'r')

        #make raw numpy array
        try:
            self.raw_npy = np.load(Path(self.path, 'raw.npy'), mmap_mode = 'r')
        except FileNotFoundError:
            if build_raw_npy:
                print('builing raw npy in ' + str(self.path))#TODO: Implement this
                
                mla.recording_to_npy(self.raw_data_path, save_name = Path(self.path,'raw'))
                self.raw_npy = np.load(Path(self.path, 'raw.npy'), mmap_mode = 'r')

        #get info on the assay
        try:
            self.assay_number = int(self.path.parts[-1]) #should this be an int or string?
            self.assay_type = self.path.parts[-2]
            self.chip_number = self.path.parts[-3]
            self.date = datetime.strptime(self.path.parts[-4], '%y%m%d').date()
            self.project = self.path.parts[-5]
        except IndexError:
            pass

    def __str__(self):
        string = ''
        string += ('path: ' + str(self.path) + '\n')
        string += ('project: ' + str(self.project) + '\n')
        string += ('date: ' + str(self.date) + '\n')
        string += ('chip_number: ' + str(self.chip_number) + '\n')
        string += ('assay_type: ' + str(self.assay_type) + '\n')
        string += ('assay_number: ' + str(self.assay_number) + '\n')

        return string

    def spike_array_from_file(self, save = True, save_name = None, **kwargs):
        """
        Runs load_spikes_from_file() and then bin_spike_data() on the result. See those for documentation on parameters.
        Returns a sparse numpy array with one axis as time and the other axis as channels with data on the spikes that occur within each time bin.
        """
        #TODO: make the savename correct
        arr = mla.spike_array_from_file(self.path, save, save_name, **kwargs)

        return arr
    #TODO: Expand the functionality of Assay to mimic what is possible in mla like I did for spike_array_from_file()


class NetworkAssay(Assay):
    pass
class ActivityScanAssay(Assay):
    pass
class StimulationAssay(Assay):
    pass



#First, find the scans in the filetree.

#The filestream goes 'project_name/date/chip_num/scan_type/assay_number
#The resulting folder has a file 'data.raw.h5' that has the raw data
#as well as an 'Analysis' folder.



parent_folder = 'D:/'
project_name = 'Summer_2023_Batch_2'

search_folder = Path(parent_folder, project_name)

#a list of chips
chips = list()
chips = ['20439', '20551']

all_h5_files = list(search_folder.glob("**/data.raw.h5"))

if not chips:
    all_network_scans = list(search_folder.glob("*/*/Network/*/data.raw.h5"))
else:
    all_network_scans = dict()
    for chip in chips:
        all_network_scans[chip] = list(NetworkAssay(raw_data.parent, True, True) for raw_data in search_folder.glob("*/" + str(chip) + "/Network/*/data.raw.h5"))
#print(all_h5_files)



print(all_network_scans)

print(all_network_scans['20439'][1])
print(list(all_network_scans['20439'][1].path.glob('*.*')))




