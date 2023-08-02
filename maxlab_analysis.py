import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, MovieWriter


#import tables
import h5py
import numpy as np
import pandas as pd
import time
import math

from scipy.signal import find_peaks
import scipy.stats as stats

from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler

"""
This script shows how to open a raw data file and how to read and interpret the data. 
ATTENTION: The data file format is not considered stable and may change in the future.
"""


def load_from_file_by_frames(filename,  start_frame, block_size, well_no = 0, recording_no = 0,  frames_per_sample = 1):
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


def load_from_file(filename, start_time, end_time,  well_no = 0, recording_no = 0, sample_frequency = 20000):
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


def recording_to_csv(filename, well_no = 0, recording_no = 0, block_size = 40000, frames_per_sample = 6, csv_name = None, delimiter = ','):
    #get channel numbers, number of frames
    #test
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        groups = h5_object['groups']
        group0 = groups[next(iter(groups))]
        
        (num_channels, num_frames) = np.shape(group0['raw'])

    column_headers = ['time'] + list(np.arange(1, num_channels + 1).astype(str))

    if csv_name == None:
        csv_name = 'data/' + filename + '.csv'
    
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

def recording_to_npy(filename, well_no = 0, recording_no = 0,  block_size = 40000, frames_per_sample = 16, save_name = None, delimiter = ','):
    #get channel numbers, number of frames
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]
        groups = h5_object['groups']
        group0 = groups[next(iter(groups))]
        
        (num_channels, num_frames) = np.shape(group0['raw'])

    if save_name == None:
        save_name = 'data/' + filename
    
    arr = np.zeros((int(num_frames/frames_per_sample), num_channels + 1), 'float32')
    
    for i, block_start in enumerate(np.arange(0, num_frames, block_size * frames_per_sample)): 
        #block_end = min(block_start + block_size * frames_per_sample, num_frames)
        frames_to_end = num_frames - block_start
        print('writing frames ' + str(block_start) + ' to '  + str(block_start + min(block_size * frames_per_sample, frames_to_end)) + ' out of ' + str(num_frames))
        X, t = load_from_file_by_frames(filename, well_no, recording_no, block_start, min(block_size, frames_to_end//frames_per_sample), frames_per_sample)
        full_arr = np.hstack((np.reshape(t, (-1, 1)), X))
        del X, t
        arr[i * block_size:i * block_size + min(block_size, frames_to_end//frames_per_sample), :] = full_arr
        
        # with open(csv_name, 'a') as csvfile:
        #     np.savetxt(csvfile, full_arr, delimiter = delimiter, fmt = '%.6g')

    np.save(save_name, arr)

def load_spikes_from_file(filename, well_no = 0, recording_no = 0, voltage_threshold = None, **kwargs):
    '''
    Returns a pd dataset of the spike data
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
    
def bin_spike_data(spike_df: pd.DataFrame, bin_size = 0.02, mode = 'binary', **kwargs):#TODO: TEST THIS!!!
    '''
    Takes in a spike data dataframe (created by load_spikes_from_file()) and turns it into a sparse numpy arrayReturns a sparse numpy array with data on the spikes that occur within each time bin.. mode must be 'binary' or 'count'. bin_size is in s.
    '''

    assert (mode == 'binary' or mode == 'count'), "mode must be binary or count."
    last_spike_t = max(spike_df['time'])
    num_channels = max(spike_df['channel'] + 1) #channels are zero-indexed
    arr = np.zeros((int(last_spike_t/bin_size + 1), num_channels), dtype=int)

    bin_start_t = 0
    i = 0
    num_spikes_lost = 0
    while bin_start_t <= last_spike_t:
        sub_df = spike_df[(spike_df['time'] >= bin_start_t) & (spike_df['time'] < bin_start_t + bin_size)]
        for channel_number in sub_df['channel']:
            if mode == 'binary':
                if arr[i, channel_number] == 1:
                    num_spikes_lost += 1
                arr[i, channel_number] = 1
            elif mode == 'count':
                arr[i, channel_number] += 1
        bin_start_t += bin_size
        i += 1 
    print('num spikes lost: ' + str(num_spikes_lost) + "/" + str(len(spike_df.index)) + '=' + str(num_spikes_lost/len(spike_df.index)))

    return arr

def spike_array_from_file(filename, save = True, save_name = None, **kwargs):
    """
    Runs load_spikes_from_file() and then bin_spike_data() on the result. See those for documentation on parameters.
    Returns a sparse numpy array with one axis as time and the other axis as channels with data on the spikes that occur within each time bin.
    """
    spike_df = load_spikes_from_file(filename, **kwargs)
    arr = bin_spike_data(spike_df, **kwargs)
    if save:
        if save_name == None:
            save_name = filename + '.binned_spikes'

    np.save(save_name, arr)

    return arr


def find_synchronized_spikes(df: pd.DataFrame, delta_t = 0.05, fraction_threshold = None, threshold_std_multiplier = 4, plot_firing = False): #TODO: make fraction threshold dependent upon distribution of numbers of neurons firing in a certain time delta
    '''
    Takes in a pd dataframe of spike data, a percentage threshold for spikes to be considered "synchronized", and a time delta (measured in seconds) in which to search for synchronized spikes.
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





def animate_pca(filestem, start_time, end_time, animation_framerate = 10, recording_framerate = 1250, speed_multiplier = 1, points_per_animation_frame = None, data_source = None, save_gif = True, save_name = None, reduce_memory_usage = False):
    '''
    Animates the first 3 axes of pca.
    filestem is the part of the data source before '.data.raw.h5'.
    start_time and end_time are in seconds. 
    animation_framerate and recording_framerate are in Hz.
    points_per_animation_frame defaults to, and maxes out at, recording_framerate * speed_multiplier / animation_framerate. 
    The passed-in value must be equal to this maximum points_per_animation frame value divided by an integer.
    data_source defaults to filestem + '.data.raw.h5'.
    save_name defaults to filestem + "_animation_" + str(start_time) + "-" + str(end_time) + "s_" + str(speed_multiplier) + "x_speed_" + str(points_per_animation_frame) + "_pts_per_" + str(animation_framerate) + "s"
    '''

    recording_frames_per_animation_frame = recording_framerate * speed_multiplier / animation_framerate

    if points_per_animation_frame == None:
        points_per_animation_frame = recording_frames_per_animation_frame

    recording_frames_per_animation_frame_subsample_rate = int(recording_frames_per_animation_frame/points_per_animation_frame)
    

    assert math.isclose(recording_frames_per_animation_frame_subsample_rate, recording_frames_per_animation_frame/points_per_animation_frame), "(recording_framerate * speed_multiplier / animation_framerate) / points_per_animation_frame must be an integer. \n (recording_framerate * speed_multiplier / animation_framerate) = " + str(recording_frames_per_animation_frame)

    if data_source == None: 
        data_source = filestem + ".data.raw.h5"
    if save_name == None:
        save_name = 'animations/' + filestem + "_animation_" + str(start_time) + "-" + str(end_time) + "s_" + str(speed_multiplier) + "x_speed_" + str(points_per_animation_frame) + "_pts_per_" + str(animation_framerate) + "s"

    data_from_npy = np.load(data_source + '.npy', mmap_mode = 'r', )

    #scale data
    t = data_from_npy[:, 0]

    if reduce_memory_usage:
        X = data_from_npy[:, 1::5]
    else:
        X = data_from_npy[:, 1::]
    Y = load_spikes_from_file(data_source, 0, 0, -10)
    Y_synchronized, spike_times = find_synchronized_spikes(Y)


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_components = 6

    pca = PCA(n_components)
    X_pca = pca.fit_transform(X_scaled)

    pc1_lims = [np.min(X_pca[:, 0]), np.max(X_pca[:, 0])]
    pc2_lims = [np.min(X_pca[:, 1]), np.max(X_pca[:, 1])]
    pc3_lims = [np.min(X_pca[:, 2]), np.max(X_pca[:, 2])]

    start_time_as_frame = start_time * recording_framerate
    end_time_as_frame = end_time * recording_framerate


    def animate_func(num):
        for ax in ax_dict.values():
            ax.clear()

        ax_dict['a'].set_xlim(pc1_lims)
        ax_dict['a'].set_ylim(pc2_lims)
        ax_dict['b'].set_xlim(pc1_lims)
        ax_dict['b'].set_ylim(pc3_lims)
        ax_dict['c'].set_xlim(pc1_lims)
        ax_dict['c'].set_ylim(pc3_lims)

        ax_dict['a'].set_xlabel('Principal component 1')
        ax_dict['a'].set_ylabel('Principal component 2')

        ax_dict['b'].set_xlabel('Principal component 1')
        ax_dict['b'].set_ylabel('Principal component 3')

        ax_dict['c'].set_xlabel('Principal component 2')
        ax_dict['c'].set_ylabel('Principal component 3')

        s1 = ax_dict['a'].scatter(X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 0], X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 1], c = t[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate], s = 2, alpha = 0.5, vmin = t[start_time_as_frame], vmax = t[end_time_as_frame])


        s2 = ax_dict['b'].scatter(X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 0], X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 2], c = t[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate], s = 2, alpha = 0.5, vmin = t[start_time_as_frame], vmax = t[end_time_as_frame])

        s3 = ax_dict['c'].scatter(X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 1], X_pca[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate, 2], c = t[start_time_as_frame:num:recording_frames_per_animation_frame_subsample_rate], s = 2, alpha = 0.5, vmin = t[start_time_as_frame], vmax = t[end_time_as_frame])



        ax_dict['z'].scatter(Y[(Y['time'] < end_time) & (Y['time'] >= start_time)]['time'], Y[(Y['time'] < end_time) & (Y['time'] >= start_time)]['channel'], 0.5, c = Y[(Y['time'] < end_time) & (Y['time'] >= start_time)]['time'])
        #plt.scatter(Y_synchronized['frameno'], Y_synchronized['channel'], 1, 'r')
        ax_dict['z'].set_xlabel('Time (s)')
        ax_dict['z'].set_ylabel('Channels')
        ax_dict['z'].vlines(spike_times[(spike_times['time'] < end_time) & (spike_times['time'] >= start_time)]['time'], 0, max(Y['channel']), 'red', alpha=0.5)
        current_time = num/recording_framerate
        #s4 = ax_dict['z'].scatter(Y[Y['time'] < current_time]['time'], Y[Y['time'] < current_time]['channel'], 0.5, c = Y[Y['time'] < current_time]['time'])
        l = ax_dict['z'].vlines(current_time, 0, max(Y['channel']), 'green')

        plt.tight_layout()

        return s1, s2, s3, l#, s4

    fig = plt.figure(figsize = (12, 8))

    ax_dict = fig.subplot_mosaic(
        """
        abc
        zzz
        """
    )


    all_animation_frames = np.arange(start_time_as_frame, end_time_as_frame, recording_frames_per_animation_frame, dtype = int)

    title = fig.suptitle('Principal axes')

    animation = FuncAnimation(fig, animate_func, interval = 1000/animation_framerate, frames = all_animation_frames, blit = True, repeat = False)

    #plt.show()

    if save_gif:
        save_start_time = time.time()
        animation.save(save_name + '.gif', writer = PillowWriter(fps=60))
        print('gif saved')
        print('time taken: ' + str(time.time() - save_start_time))
    
    return fig


if __name__ == "__main__":
    pass
