import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
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
    Returns a tuple of (spike_times, channels, magnitudes, threshold)
    '''
    with h5py.File(filename, "r") as h5_file:
        h5_object = h5_file['wells']['well{0:0>3}'.format(well_no)]['rec{0:0>4}'.format(recording_no)]

        # Load settings from file
        threshold = h5_object['settings']['spike_threshold'][0]


        #first_frame_no = h5_object['groups']['routed']['frame_nos'][0]

        spike_dataset = h5_object['spikes']
        
        spike_array = np.array(spike_dataset)
        spike_pd_dataset = pd.DataFrame(spike_array)
        
        
        first_frame_no = spike_pd_dataset['frameno'][0]

        spike_pd_dataset['frameno'] = spike_pd_dataset['frameno'] - first_frame_no

        if voltage_threshold == None:
            return spike_pd_dataset
        else:
            pass



    #return (spike_times, channels, magnitudes, threshold)





if __name__ == "__main__":
    filename = 'div21.data.raw.h5'

    Y = load_spikes_from_file(filename, 0, 0)
    print(Y.head())

    amp_thresh = -10

    # This code doesn't work but you need to filter out the spikes that are noise
    Y = Y.loc[Y['amplitude'].le(amp_thresh)]

    print(Y.head())

    # Now we need to look at the data with your eyeballs
    plt.scatter(Y['frameno']/20000, Y['channel'], 1)

    plt.show()

    # X,t = load_from_file(filename, 0, 0, 1, 20, 800)
    # print(np.shape(X))

    # plt.plot(t,X[:,:]);
    # plt.ylabel('Volts');
    # plt.xlabel('Seconds');


    # plt.show()



    # plt.savefig('plot.png')
