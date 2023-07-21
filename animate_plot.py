import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, MovieWriter


import h5py
#from read_raw import load_from_file
import maxlab_analysis as mla

import time
from scipy.signal import find_peaks
import scipy.stats as stats
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler

filestem = 'batch2_20647div14'

filename = filestem + ".data.raw.h5"

start_time = 100
end_time = 200
step = 0.1
speed_multiplier = 5
framerate = 1250
points_per_time_step = 100 #this maxes out at step*framerate*speed_multiplier
points_per_time_step = step*framerate*speed_multiplier
save_name = filestem + "_animation_" + str(start_time) + "-" + str(end_time) + "s_" + str(speed_multiplier) + "x_speed"






data_from_npy = np.load(filename + '.npy', mmap_mode = 'r', )
#scale data

t = data_from_npy[:, 0]
X = data_from_npy[:, 1::]



Y = mla.load_spikes_from_file(filename, 0, 0, -10)
print(np.shape(Y))

timer_start = time.time()
Y_synchronized, spike_times = mla.find_synchronized_spikes(Y)
timer_end = time.time()
print('time taken: ' + str(timer_end - timer_start) + ' s')
print(Y_synchronized.head())
print(spike_times.head())

plt.figure(figsize = (10, 10))
plt.scatter(Y['time'], Y['channel'], 0.5)
#plt.scatter(Y_synchronized['time'], Y_synchronized['channel'], 1, 'r')
plt.xlabel('Time (s)')
plt.ylabel('Channels')
plt.vlines(spike_times['time'], 0, max(Y['channel']), 'red', alpha=0.3)




scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Plot a subset of the channels
plt.figure(figsize = (10, 5))
plt.subplot(121)
plt.plot(t[:], X[:, ::10], linewidth = 0.5)
plt.title('pre scaling')

plt.subplot(122)
plt.plot(t[:], X_scaled[:, ::10], linewidth = 0.5)
plt.title('post scaling')
plt.show()


n_components = 6

pca = PCA(n_components)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)




pc1_lims = [np.min(X_pca[:, 0]), np.max(X_pca[:, 0])]
pc2_lims = [np.min(X_pca[:, 1]), np.max(X_pca[:, 1])]
pc3_lims = [np.min(X_pca[:, 2]), np.max(X_pca[:, 2])]

start_time_as_frame = start_time * framerate


fig_width = 3
plt.figure(figsize = (fig_width * 5, (int(np.ceil(n_components/fig_width)) * 5)))
for i in np.arange(1, n_components + 1):
    plt.subplot(int(np.ceil(n_components/fig_width)), fig_width, i)
    plt.plot(t, X_pca[:, i-1], linewidth = 0.5, alpha = 0.9)
    plt.title('Principal component ' + str(i))
plt.tight_layout()
plt.show()


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

    index_step = step * framerate * speed_multiplier
    s1 = ax_dict['a'].scatter(X_pca[start_time_as_frame:num:int(index_step/points_per_time_step), 0], X_pca[start_time_as_frame:num:int(index_step/points_per_time_step), 1], c = t[start_time_as_frame:num:int(index_step/points_per_time_step)], s = 2, alpha = 0.5, vmin = t[start_time_as_frame], vmax = t[end_time * framerate])


    s2 = ax_dict['b'].scatter(X_pca[start_time_as_frame:num:int(index_step/points_per_time_step), 0], X_pca[start_time_as_frame:num:int(index_step/points_per_time_step), 2], c = t[start_time_as_frame:num:int(index_step/points_per_time_step)], s = 2, alpha = 0.5, vmin = t[start_time_as_frame], vmax = t[end_time * framerate])

    s3 = ax_dict['c'].scatter(X_pca[start_time_as_frame:num:int(index_step/points_per_time_step), 1], X_pca[start_time_as_frame:num:int(index_step/points_per_time_step), 2], c = t[start_time_as_frame:num:int(index_step/points_per_time_step)], s = 2, alpha = 0.5, vmin = t[start_time_as_frame], vmax = t[end_time * framerate])



    ax_dict['z'].scatter(Y[(Y['time'] < end_time) & (Y['time'] >= start_time)]['time'], Y[(Y['time'] < end_time) & (Y['time'] >= start_time)]['channel'], 0.5, c = Y[(Y['time'] < end_time) & (Y['time'] >= start_time)]['time'])
    #plt.scatter(Y_synchronized['frameno'], Y_synchronized['channel'], 1, 'r')
    ax_dict['z'].set_xlabel('Time (s)')
    ax_dict['z'].set_ylabel('Channels')
    ax_dict['z'].vlines(spike_times[(spike_times['time'] < end_time) & (spike_times['time'] >= start_time)]['time'], 0, max(Y['channel']), 'red', alpha=0.5)
    current_time = num/framerate
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


all_frames = np.arange(start_time_as_frame, end_time * framerate, step * framerate * speed_multiplier, dtype = int)

title = fig.suptitle('Principal axes')

animation = FuncAnimation(fig, animate_func, interval = step * 1000, frames = all_frames, blit = True, repeat = False)

plt.show()

save_start_time = time.time()
animation.save('data/' + save_name + '.gif', writer = PillowWriter(fps=60))
print('gif saved')
print('time taken: ' + str(time.time() - save_start_time))
