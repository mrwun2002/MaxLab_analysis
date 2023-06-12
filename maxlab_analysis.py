import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import time

from scipy.signal import find_peaks

"""
This script shows how to open a raw data file and how to read and interpret the data. 
ATTENTION: The data file format is not considered stable and may change in the future.
"""


def load_from_file(filename, well_no, recording_no, start_frame, block_size):
    # The maximum allowed block size can be increased if needed,
    # However, if the block size is too large, handling of the data in Python gets too slow.

    #sample rate is 20000 samples per second
    max_allowed_block_size = 40000
    assert(block_size<=max_allowed_block_size)
    h5_file = h5py.File(filename, 'r')
    h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]

    # Load settings from file
    lsb = h5_object['settings']['lsb'][0]
    sampling = h5_object['settings']['sampling'][0]

    # compute time vector
    #stop_frame = start_frame+block_size
    time = np.arange(start_frame,start_frame+block_size) / 20e3

    # Load raw data from file
    groups = h5_object['groups']
    print(groups)
    group0 = groups[next(iter(groups))]

    return group0['raw'][:,start_frame:start_frame+block_size].T * lsb , time



def load_from_file(filename, well_no, recording_no, start_time, end_time, sample_frequency):
    '''
    The native sample rate is 20000 samples per second.
    Returns (voltage traces, time)
    '''
    # The maximum allowed block size can be increased if needed,
    # However, if the block size is too large, handling of the data in Python gets too slow.

    #sample rate is 20000 samples per second
    start_frame = round(start_time * 20000)
    end_frame = round(end_time * 20000)
    block_size = end_frame - start_frame


    frames_per_sample = round(20000/sample_frequency)

    max_allowed_block_size = 40000
    
    assert(block_size//frames_per_sample <=max_allowed_block_size)

    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]

        # Load settings from file
        lsb = h5_object['settings']['lsb'][0]
        sampling = h5_object['settings']['sampling'][0]

        # compute time vector
        #stop_frame = start_frame+block_size
        time = np.arange(start_frame,start_frame+block_size, frames_per_sample) / 20e3

        # Load raw data from file
        groups = h5_object['groups']
        group0 = groups[next(iter(groups))] #there always only seems to be one group within this calle "routed" - can we replace this code to be more specific?

        return group0['raw'][:,start_frame:start_frame+block_size:frames_per_sample].T * lsb , time



def load_spikes_from_file(filename, well_no, recording_no, voltage_threshold = None):
    '''
    Returns a pd dataset of the spike data)
    '''
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]

        # Load settings from file
        sd_threshold = h5_object['settings']['spike_threshold'][0]


        #first_frame_no = h5_object['groups']['routed']['frame_nos'][0]

        spike_dataset = h5_object['spikes']
        
        spike_array = np.array(spike_dataset)
        spike_pd_dataset = pd.DataFrame(spike_array)
        
        
        first_frame_no = spike_pd_dataset['frameno'][0]

        spike_pd_dataset['frameno'] = spike_pd_dataset['frameno'] - first_frame_no

        if voltage_threshold != None:
            spike_pd_dataset = spike_pd_dataset.loc[spike_pd_dataset['amplitude'].le(voltage_threshold)]
        
        return spike_pd_dataset

def find_synchronized_spikes(df: pd.DataFrame, delta_frame = 1000, fraction_threshold = None, threshold_std_multiplier = 4, plot_firing = False): #TODO: make fraction threshold dependent upon distribution of numbers of neurons firing in a certain time delta
    '''
    Takes in a pd dataframe of spike data, a percentage threshold for spikes to be considered "synchronized", and a time delta (measured in frames) in which to search for synchronized spikes.
    Returns a pd dataframe containing just the synchronized spikes as well as a dataframe containin the start times of each spike, to the lowest time delta divided by two, and the number of channels active during each spike.
    '''
    num_channels = df['channel'].nunique() #Do we want this to be the number of active channels? the largest channel number?

    max_frameno = max(df['frameno'])

    synchronized_spikes_df = pd.DataFrame(columns=df.columns)

    #look within a range of framenos, if a spike exists within that range, add the channel to a set of channels. Count the number of elements in that set.
    
    #for start_frame in df['frameno'].unique():

    start_frames = np.arange(0, max_frameno, delta_frame//2)
    fraction_firing_channels = np.zeros_like(start_frames, dtype=float)        


    for i, start_frame in enumerate(start_frames):

        entries_in_range = df.loc[df['frameno'].ge(start_frame) & df['frameno'].lt(start_frame + delta_frame)]

        fraction_firing_channels[i] = (entries_in_range['channel'].nunique())/num_channels
        #if fraction_firing_channels[i] >= fraction_threshold:
            #synchronized_spikes_df = pd.concat([synchronized_spikes_df, entries_in_range])
            #spike_times.append(start_frame)

    if fraction_threshold == None:
        fraction_threshold = threshold_std_multiplier*np.std(fraction_firing_channels) + np.mean(fraction_firing_channels)
        #print(fraction_threshold)
    
    (spike_indeces, spike_properties) = find_peaks(fraction_firing_channels, height = fraction_threshold)
    #print(spike_indeces)
    spike_times = start_frames[spike_indeces]

    spike_times_df = pd.DataFrame(list(zip(spike_times, spike_properties['peak_heights'])), columns = ['time', 'fraction channels active'])

    if plot_firing:
        plt.figure()
        plt.plot(start_frames, fraction_firing_channels)
        plt.xlabel('Frame number')
        plt.ylabel('Number of channels')
        plt.hlines(fraction_threshold, 0, max_frameno, 'red')
        #plt.show()

    for spike_time in spike_times:
        entries_in_range = df.loc[df['frameno'].ge(spike_time) & df['frameno'].lt(spike_time + delta_frame)]
        synchronized_spikes_df = pd.concat([synchronized_spikes_df, entries_in_range])

    return synchronized_spikes_df, spike_times_df





if __name__ == "__main__":
    filename = 'div21.data.raw.h5'

    Y = load_spikes_from_file(filename, 0, 0)
    #print(Y.head())

    amp_thresh = -10

    # This code doesn't work but you need to filter out the spikes that are noise
    Y = Y.loc[Y['amplitude'].le(amp_thresh)]

    #print(Y.head())

    start_time = time.time()
    Y_synchronized, spike_times = find_synchronized_spikes(Y, plot_firing = True)
    end_time = time.time()
    print('time taken: ' + str(end_time - start_time) + ' s')
    print(Y_synchronized.head())
    print(spike_times.head())
    #print(len(spike_times))

    # Now we need to look at the data with your eyeballs
    plt.figure()
    plt.scatter(Y['frameno'], Y['channel'], 1)
    #plt.scatter(Y_synchronized['frameno'], Y_synchronized['channel'], 1, 'r')
    plt.xlabel('Frame number')
    plt.ylabel('Channels')
    plt.vlines(spike_times['time'], 0, max(Y['channel']), 'red', alpha=0.3)


    #get differences between spike times
    spike_diffs = spike_times.diff()
    print(spike_diffs.head())
    plt.figure()
    plt.scatter(spike_diffs['time'], spike_diffs['fraction channels active'])


    #histogram of spike diffs
    plt.hist(spike_diffs['time'])
    plt.show()

    # X,t = load_from_file(filename, 0, 0, 1, 20, 800)
    # print(np.shape(X))

    # plt.plot(t,X[:,:]);
    # plt.ylabel('Volts');
    # plt.xlabel('Seconds');


    # plt.show()



    # plt.savefig('plot.png')
