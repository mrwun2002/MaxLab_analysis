import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import h5py
#from read_raw import load_from_file
import maxlab_analysis as mla
from assay import *

import time
from scipy.signal import find_peaks
import scipy.stats as stats
from scipy.spatial import distance

from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.cluster import KMeans, DBSCAN, OPTICS, AgglomerativeClustering#SpectralClustering#, HDBSCAN
import seaborn as sns
from umap import UMAP
import umap.plot

from pathlib import Path
from datetime import datetime

import pprint



def analyze(assay: Assay):
    #First, design the functions to build each numpy array. 
    #The first few are commented out bc they're done on initialization
    #build_spike_array_func = lambda filename: mla.bin_spike_data(assay.spike_df, bin_size = 0.02, mode = 'binary', save = True, save_name = Path(assay.path, filename))
    #build_raw_npy_func = lambda filename: mla.recording_to_npy(assay.raw_data_path, well_no = 0, recording_no = 0,  block_size = 40000, frames_per_sample = 16, save_name = Path(assay.path, filename))
    def build_scaled_func(full_filename):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(assay.analyses['raw'][:, 1:])
        full_arr = np.hstack([assay.analyses['raw'][:, 0].reshape(-1, 1), X_scaled])

        np.save(full_filename, full_arr)
    
    def build_pca_func(full_filename):
        #50 components, operates on scaled data
        pca = PCA(50)
        X_pca = pca.fit_transform(assay.analyses['scaled'][:, 1:])
        full_arr = np.hstack([assay.analyses['scaled'][:, 0].reshape(-1, 1), X_pca])

        np.save(full_filename, full_arr)

    #actual analysis
    print('starting analysis')
    assay.load_build_npy('scaled', build_scaled_func)
    assay.load_build_npy('pca', build_pca_func)

    full_channel_synchronized, burst_times = mla.find_synchronized_bursts(assay.spike_df, plot_firing = True)
    print(burst_times.head())

    burst_frequency = (len(burst_times))/assay.record_time
    print(burst_frequency)

    print(assay)
    burst_frequency_df.loc[len(burst_frequency_df.index)] = ([assay.chip_number, assay.date, assay.assay_number, burst_frequency])    
        
    

parent_folder = 'q:/Users/mrwun/OneDrive - Yale University/levin'
parent_folder = 'D:/'

project_name = 'Summer_2023_Batch_2'

all_network_scans = mla.load_assays_from_project(parent_folder, project_name)

pprint.pprint(all_network_scans)

#assay = all_network_scans['20551'][0]

burst_frequency_df = pd.DataFrame(columns = ['chip_number', 'date', 'assay_number', 'burst_frequency'])

print(all_network_scans.values())
for assay in np.concatenate(list(all_network_scans.values())):
    analyze(assay)

print(burst_frequency_df)

burst_frequency_df.to_csv('burst_frequency.csv')
