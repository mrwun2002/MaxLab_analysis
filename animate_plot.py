import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


import h5py
#from read_raw import load_from_file
import maxlab_analysis as mla

import time
from scipy.signal import find_peaks
import scipy.stats as stats
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler

filename = "div21.data.raw.h5"

data_from_npy = np.load(filename + '.npy', mmap_mode = 'r', )
#scale data

t = data_from_npy[:, 0]
X = data_from_npy[:, 1::2]



Y = mla.load_spikes_from_file(filename, 0, 0, -10)
print(np.shape(Y))

start_time = time.time()
Y_synchronized, spike_times = mla.find_synchronized_spikes(Y)
end_time = time.time()
print('time taken: ' + str(end_time - start_time) + ' s')
print(Y_synchronized.head())
print(spike_times.head())

plt.figure(figsize = (10, 10))
plt.scatter(Y['time'], Y['channel'], 0.5)
#plt.scatter(Y_synchronized['frameno'], Y_synchronized['channel'], 1, 'r')
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






end_time = 10
step = 0.1
speed_multiplier = 1

pc1_lims = [np.min(X_pca[:, 0]), np.max(X_pca[:, 0])]
pc2_lims = [np.min(X_pca[:, 1]), np.max(X_pca[:, 1])]
pc3_lims = [np.min(X_pca[:, 2]), np.max(X_pca[:, 2])]





def animate_func(num):
    #ax.clear()
    index_step = int(step * 1250 * speed_multiplier)
    s1 = ax_dict['a'].scatter(X_pca[0:num, 0], X_pca[0:num, 1], c = t[0:num], s = 2, alpha = 0.5)


    s2 = ax_dict['b'].scatter(X_pca[0:num, 0], X_pca[0:num, 2], c = t[0:num], s = 2, alpha = 0.5)

    current_time = num/1250/speed_multiplier
    s3 = ax_dict['c'].scatter(Y[Y['time'] < current_time]['time'], Y[Y['time'] < current_time]['channel'], 0.5, cmap = Y[Y['time'] < current_time]['time'])
    l = ax_dict['c'].vlines(current_time, 0, max(Y['channel']), 'green')

    title = fig.suptitle('Principal axes')
    return s1, s2, s3, l

fig = plt.figure(figsize = (10, 10))

ax_dict = fig.subplot_mosaic(
    """
    ab
    cc
    """
)
ax_dict['a'].set_xlabel('Principal component 1')
ax_dict['a'].set_ylabel('Principal component 2')
ax_dict['a'].set_xlim(pc1_lims)
ax_dict['a'].set_ylim(pc2_lims)

ax_dict['b'].set_xlabel('Principal component 1')
ax_dict['b'].set_ylabel('Principal component 3')
ax_dict['b'].set_xlim(pc1_lims)
ax_dict['b'].set_ylim(pc3_lims)

ax_dict['c'].scatter(Y[Y['time'] < end_time]['time'], Y[Y['time'] < end_time]['channel'], 0.5)
#plt.scatter(Y_synchronized['frameno'], Y_synchronized['channel'], 1, 'r')
ax_dict['c'].set_xlabel('Time (s)')
ax_dict['c'].set_ylabel('Channels')
ax_dict['c'].vlines(spike_times[spike_times['time'] < end_time]['time'], 0, max(Y['channel']), 'red', alpha=0.3)



animation = FuncAnimation(fig, animate_func, interval = step * 1000, frames = np.arange(0, end_time * 1250, step * 1250 * speed_multiplier, dtype = int), blit = True)
plt.show()
