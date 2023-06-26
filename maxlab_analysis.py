import matplotlib.pyplot as plt
#import tables
import h5py
import numpy as np
import pandas as pd
import time

from scipy.signal import find_peaks
import scipy.stats as stats

"""
This script shows how to open a raw data file and how to read and interpret the data. 
ATTENTION: The data file format is not considered stable and may change in the future.
"""


def load_from_file_by_frames(filename, well_no, recording_no, start_frame, block_size, frames_per_sample = 1):
    # The maximum allowed block size can be increased if needed,
    # However, if the block size is too large, handling of the data in Python gets too slow.

    #sample rate is 20000 samples per second
    max_allowed_block_size = 40000
    assert(block_size<=max_allowed_block_size)

    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]

        # Load settings from file
        lsb = h5_object['settings']['lsb'][0]
        sampling = h5_object['settings']['sampling'][0]

        # compute time vector
        #stop_frame = start_frame+block_size
        time = np.arange(start_frame,start_frame+block_size * frames_per_sample,frames_per_sample) / 20e3

        # Load raw data from file
        groups = h5_object['groups']
        #print(groups)
        group0 = groups[next(iter(groups))]

        try:
            return group0['raw'][:,start_frame:start_frame+block_size*frames_per_sample:frames_per_sample].T * lsb , time
        except OSError:
            print("OSError thrown")
            num_channels = np.shape(group0['raw'])[0]
            placeholder_data = np.tile(np.arange(start_frame, start_frame + block_size*frames_per_sample, frames_per_sample), (num_channels, 1)).T
            return placeholder_data, time


def load_from_file(filename, well_no, recording_no, start_time, end_time, sample_frequency = 20000):
    '''
    The native sample rate is 20000 samples per second.
    Returns (voltage traces, time)
    '''
    # The maximum allowed block size can be increased if needed,
    # However, if the block size is too large, handling of the data in Python gets too slow.

    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]

        # Load settings from file
        lsb = h5_object['settings']['lsb'][0]
        native_sampling_rate = h5_object['settings']['sampling'][0]

        #sample rate is 20000 samples per second
        start_frame = round(start_time * native_sampling_rate)
        end_frame = round(end_time * native_sampling_rate)
        block_size = end_frame - start_frame


        frames_per_sample = round(native_sampling_rate/sample_frequency)

        max_allowed_block_size = 40000
        
        assert(block_size//frames_per_sample <=max_allowed_block_size)

        # compute time vector
        #stop_frame = start_frame+block_size
        time = np.arange(start_frame,start_frame+block_size, frames_per_sample) / native_sampling_rate

        # Load raw data from file
        groups = h5_object['groups']
        group0 = groups[next(iter(groups))] #there always only seems to be one group within this calle "routed" - can we replace this code to be more specific?
        
        try:
            return group0['raw'][:,start_frame:start_frame+block_size:frames_per_sample].T * lsb , time
        except OSError:
            print("OSError thrown")
            num_channels = np.shape(group0['raw'])[0]
            placeholder_data = np.tile(np.arange(start_frame, start_frame + block_size, frames_per_sample), (num_channels, 1)).T
            return placeholder_data, time


def recording_to_csv(filename, well_no, recording_no, block_size = 40000, frames_per_sample = 6, csv_name = None, delimiter = ','):
    #get channel numbers, number of frames
    #test
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        groups = h5_object['groups']
        group0 = groups[next(iter(groups))]
        
        (num_channels, num_frames) = np.shape(group0['raw'])

    column_headers = ['time'] + list(np.arange(1, num_channels + 1).astype(str))

    if csv_name == None:
        csv_name = filename + '.csv'
    
    #df = pd.DataFrame(columns = column_headers)
    #df.to_csv(csv_name, mode = 'w', index = False, header = True)

    with open(csv_name, 'w') as csvfile:
        np.savetxt(csvfile, [], header = ' '.join(column_headers), delimiter=delimiter)

    
    for block_start in np.arange(0, num_frames, block_size * frames_per_sample): 
        #block_end = min(block_start + block_size * frames_per_sample, num_frames)
        frames_to_end = num_frames - block_start
        print('writing frames ' + str(block_start) + ' to '  + str(block_start + min(block_size * frames_per_sample, frames_to_end)) + ' out of ' + str(num_frames))
        X, t = load_from_file_by_frames(filename, well_no, recording_no, block_start, min(block_size, frames_to_end//frames_per_sample), frames_per_sample)
        full_arr = np.hstack((np.reshape(t, (-1, 1)), X))
        with open(csv_name, 'a') as csvfile:
            np.savetxt(csvfile, full_arr, delimiter = delimiter, fmt = '%.6g')
    

def load_spikes_from_file(filename, well_no, recording_no, voltage_threshold = None):
    '''
    Returns a pd dataset of the spike data)
    '''
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]

        # Load settings from file
        sd_threshold = h5_object['settings']['spike_threshold'][0]
        native_sampling_rate = h5_object['settings']['sampling'][0]


        #first_frame_no = h5_object['groups']['routed']['frame_nos'][0]

        spike_dataset = h5_object['spikes']
        
        spike_array = np.array(spike_dataset)
        spike_pd_dataset = pd.DataFrame(spike_array)
        
        
        first_frame_no = spike_pd_dataset['frameno'][0]

        spike_pd_dataset['frameno'] = spike_pd_dataset['frameno'] - first_frame_no

        if voltage_threshold != None:
            spike_pd_dataset = spike_pd_dataset.loc[spike_pd_dataset['amplitude'].le(voltage_threshold)]

        spike_pd_dataset['frameno'] = spike_pd_dataset['frameno'].multiply(1/native_sampling_rate)
        
        spike_pd_dataset.rename(columns = {'frameno':'time'}, inplace = True)

        
        return spike_pd_dataset

def find_synchronized_spikes(df: pd.DataFrame, delta_t = 0.05, fraction_threshold = None, threshold_std_multiplier = 4, plot_firing = False): #TODO: make fraction threshold dependent upon distribution of numbers of neurons firing in a certain time delta
    '''
    Takes in a pd dataframe of spike data, a percentage threshold for spikes to be considered "synchronized", and a time delta (measured in frames) in which to search for synchronized spikes.
    Returns a pd dataframe containing just the synchronized spikes as well as a dataframe containin the start times of each spike, to the lowest time delta divided by two, and the number of channels active during each spike.
    '''
    num_channels = df['channel'].nunique() #Do we want this to be the number of active channels? the largest channel number?

    max_time = max(df['time'])

    synchronized_spikes_df = pd.DataFrame(columns=df.columns)

    #look within a range of times, if a spike exists within that range, add the channel to a set of channels. Count the number of elements in that set.
    
    start_times = np.arange(0, max_time, delta_t/2)
    fraction_firing_channels = np.zeros_like(start_times, dtype=float)        


    for i, start_time in enumerate(start_times):

        entries_in_range = df.loc[df['time'].ge(start_time) & df['time'].lt(start_time + delta_t)]

        fraction_firing_channels[i] = (entries_in_range['channel'].nunique())/num_channels
        #if fraction_firing_channels[i] >= fraction_threshold:
            #synchronized_spikes_df = pd.concat([synchronized_spikes_df, entries_in_range])
            #spike_times.append(start_frame)

    if fraction_threshold == None:
        fraction_threshold = threshold_std_multiplier*np.std(fraction_firing_channels) + np.mean(fraction_firing_channels)
        #print(fraction_threshold)
    
    (spike_indeces, spike_properties) = find_peaks(fraction_firing_channels, height = fraction_threshold)
    #print(spike_indeces)
    spike_times = start_times[spike_indeces]

    spike_times_df = pd.DataFrame(list(zip(spike_times, spike_properties['peak_heights'])), columns = ['time', 'fraction channels active'])

    if plot_firing:
        plt.figure()
        plt.plot(start_times, fraction_firing_channels)
        plt.xlabel('Time (s)')
        plt.ylabel('Number of channels')
        plt.hlines(fraction_threshold, 0, max_time, 'red')
        #plt.show()

    for spike_time in spike_times:
        entries_in_range = df.loc[df['time'].ge(spike_time) & df['time'].lt(spike_time + delta_t)]
        synchronized_spikes_df = pd.concat([synchronized_spikes_df, entries_in_range])

    return synchronized_spikes_df, spike_times_df





if __name__ == "__main__":
    filename = 'div28.data.raw.h5'
    #X, t = load_from_file(filename, 0, 0, 0.2, 10 , 2000)
    # plt.plot(t,X[:,:]);
    # plt.ylabel('Volts');
    # plt.xlabel('Seconds');


    plt.show()

    Y = load_spikes_from_file(filename, 0, 0)
    #print(Y.head())

    amp_thresh = -10

    Y = Y.loc[Y['amplitude'].le(amp_thresh)]
    print(np.shape(Y))

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
    plt.scatter(Y['time'], Y['channel'], 1)
    #plt.scatter(Y_synchronized['frameno'], Y_synchronized['channel'], 1, 'r')
    plt.xlabel('Time (s)')
    plt.ylabel('Channels')
    plt.vlines(spike_times['time'], 0, max(Y['channel']), 'red', alpha=0.3)


    #get differences between spike times
    spike_diffs = spike_times.diff().dropna()
    '''
    print(spike_diffs.head())
    plt.figure()
    plt.scatter(spike_diffs['time'], spike_diffs['fraction channels active'])
    plt.xlabel('Time (s)')
    plt.ylabel('Channels')
    '''

    #histogram of spike diffs
    num_bins = 'auto'
    #num_bins = 1000
    
    plt.figure()
    plt.subplot(231)
    plt.hist(spike_diffs['time'], num_bins, density = True)
    plt.xlabel('Interburst interval (s)')
    plt.ylabel('Probability of observation')

    print(spike_diffs['time'])
    print(1/spike_diffs['time'])

    plt.subplot(232)
    plt.hist(1/spike_diffs['time'], num_bins, density = True)
    plt.xlabel('Burst frequency (1/s)')
    plt.ylabel('Probability of observation')


    plt.subplot(233)
    plt.hist(spike_diffs['time'], num_bins, density = True, cumulative = True)
    plt.xlabel('t (s)')
    plt.ylabel('$P(IBI) < t$')
    

    plt.subplot(234)
    plt.hist(1/spike_diffs['time'], num_bins, density = True, cumulative = True)
    plt.xlabel('f (1/s)')
    plt.ylabel('$P(burst frequency) < f$')

    plt.subplot(235)
    plt.hist(stats.norm.pdf(spike_diffs['time']), num_bins, density = True)
    plt.xlabel('t (s)')
    plt.ylabel('$P(IBI) < t$')
    

    plt.subplot(236)
    plt.hist(stats.norm.pdf(1/spike_diffs['time']), num_bins, density = True, cumulative = True)
    plt.xlabel('f (1/s)')
    plt.ylabel('$P(burst frequency) < f$')

    
    ###
    #scipy stats version?
    IBI_hist = np.histogram(spike_diffs['time'], bins = num_bins, density = True)
    (IBI_data, IBI_bins) = IBI_hist
    IBI_bin_midpoints = [(a + b) /2 for a,b in zip(IBI_bins[:-1], IBI_bins[1:])]

    hist_dist = stats.rv_histogram(IBI_hist, density = True)
    X = np.linspace(0, max(spike_diffs['time']), 500)
    
    plt.figure()
    plt.subplot(131)
    plt.title('IBI distribution')
    plt.scatter(IBI_bin_midpoints, IBI_data)
    #plt.plot(X, hist_dist.pdf(X), label= 'PDF')
    #plt.plot(X, hist_dist.cdf(X), label= 'CDF')
    
    plt.legend()

    plt.xlabel('Interburst interval (s)')
    plt.ylabel('Probability of observation')
    plt.yscale('log')
    plt.xscale('log')

    plt.subplot(132)
    #NOTE - THIS NAN_TO_NUM THING IS HAPPENING
    log_IBI_bin_midpoints = np.log(IBI_bin_midpoints)
    log_IBI_data = np.log1p(IBI_data)

    fit = stats.linregress(log_IBI_bin_midpoints, log_IBI_data)
    plt.plot(log_IBI_bin_midpoints, fit.intercept + fit.slope*log_IBI_bin_midpoints, 'r', label=f'fitted line, $r^2 = {fit.rvalue**2:.2f}$')

    plt.title('IBI distribution, log/log')
    #plt.hist(spike_diffs['time'], num_bins, density = True)
    plt.plot(log_IBI_bin_midpoints, log_IBI_data, label = "log fit")
    #plt.loglog(X, hist_dist.pdf(X), label= 'PDF')
    #plt.loglog(X, hist_dist.cdf(X), label= 'CDF')
    
    plt.legend()

    plt.xlabel('log(Interburst interval (s))')
    plt.ylabel('log(Probability of observation)')

    plt.subplot(133)
    plt.title('IBI distribution, log/log')
    #plt.hist(spike_diffs['time'], num_bins, density = True)
    plt.loglog(IBI_bin_midpoints, IBI_data, label = "log fit")
    plt.loglog(X, hist_dist.pdf(X), label= 'PDF')
    plt.loglog(X, hist_dist.cdf(X), label= 'CDF')
    
    plt.legend()

    #plt.xlabel('log(Interburst interval (s))')
    #plt.ylabel('log(Probability of observation)')
    plt.show()
    # X,t = load_from_file(filename, 0, 0, 1, 20, 800)
    # print(np.shape(X))

    # plt.plot(t,X[:,:]);
    # plt.ylabel('Volts');
    # plt.xlabel('Seconds');


    # plt.show()



    # plt.savefig('plot.png')
