
from pathlib import Path
from datetime import datetime
import maxlab_analysis as mla
import numpy as np
import pandas as pd
import sys

from sklearn.preprocessing import StandardScaler



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
    def __init__(self, path, build_raw_npy = True, build_spike_array = True, overwrite_raw_npy = False, overwrite_spike_array = False, build_umap = True):
        '''
        Creates an Assay object from a path to a folder. Takes in a filepath to the folder with all data files in the form of a Path object or a string.
        the 'build' parameters tell you whether to create a file if it does not yet exist. They are either True or False.
        the 'overwrite' parameters, if true, ignore whether or not there already exists a spike array or raw npy file. build parameter must be true for this to take effect.
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
        if not overwrite_spike_array:
            try:
                self.spike_array = np.load(Path(self.path, 'spike_array.npy'), mmap_mode = 'r')
            except FileNotFoundError:
                if build_spike_array:
                    self.build_spike_array()
                    self.spike_array = np.load(Path(self.path, 'spike_array.npy'), mmap_mode = 'r')
        else:
            self.build_spike_array()
            self.spike_array = np.load(Path(self.path, 'spike_array.npy'), mmap_mode = 'r')

        #make raw numpy array
        if not overwrite_raw_npy:
            try:
                self.raw_npy = np.load(Path(self.path, 'raw.npy'), mmap_mode = 'r')
            except FileNotFoundError:
                if build_raw_npy:
                    self.build_raw_npy()
                    self.raw_npy = np.load(Path(self.path, 'raw.npy'), mmap_mode = 'r')
        else:
            self.build_raw_npy()
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

    #all of these build functions are of similar structures. THere may be a cleaner way to go about this - maybe have a class called builder that these methods inherit from or smth
    def build_spike_array(self, save_name = 'spike_array', bin_size = 0.02, mode = 'binary'):
        print('building spike array in ' + str(self.path))
        #replaces spike_df with a version that also includes bin_id of each spike.
        mla.bin_spike_data(self.spike_df, bin_size = bin_size, mode = mode, save = True, save_name = Path(self.path, save_name))

    def build_raw_npy(self, save_name = 'raw', well_no = 0, recording_no = 0,  block_size = 40000, frames_per_sample = 16 ):
        print('builing raw npy in ' + str(self.path))                
        mla.recording_to_npy(self.raw_data_path, well_no = well_no, recording_no = recording_no,  block_size = block_size, frames_per_sample = frames_per_sample, save_name = Path(self.path, save_name))

    def build_scaled_raw(self, save_name = 'scaled_raw'):
        '''
        Uses standardscaler
        '''
        print('building scaled raw in ' + str(self.path))

    def build_umap(self, save_name = 'umap'):
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
    
    def __repr__(self):
        return str(self.path)

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


if __name__ == "__main__":

    parent_folder = '/media/mxwbio/Elements'
    project_name = 'Summer_2023_Batch_2'

    all_network_scans = mla.load_assays_from_project(parent_folder, project_name)

    print(all_network_scans)

    print(all_network_scans['20439'][1])
    print(list(all_network_scans['20439'][1].path.glob('*.*')))


