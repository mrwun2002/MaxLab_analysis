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
step = 0.2

pc1_lims = [np.min(X_pca[:, 0]), np.max(X_pca[:, 0])]
pc2_lims = [np.min(X_pca[:, 1]), np.max(X_pca[:, 1])]
pc3_lims = [np.min(X_pca[:, 2]), np.max(X_pca[:, 2])]

def animate_func(num):
    #ax.clear()

    ax1.scatter(X_pca[num, 0], X_pca[num, 1], c = t[num], s = 2, alpha = 0.5)


    ax2.scatter(X_pca[num, 0], X_pca[num, 2], c = t[num], s = 2, alpha = 0.5)

    fig.suptitle('Time = ' + str(np.round(t[num] * 1000, decimals = 3)) + 'ms')

fig = plt.figure(figsize = (10, 10))
ax1 = plt.subplot(121)
ax1.set_xlabel('Principal component 1')
ax1.set_ylabel('Principal component 2')
ax1.set_xlim(pc1_lims)
ax1.set_ylim(pc2_lims)

ax2 = plt.subplot(122)
ax2.set_xlabel('Principal component 1')
ax2.set_ylabel('Principal component 3')
ax2.set_xlim(pc1_lims)
ax2.set_ylim(pc3_lims)


ax2 = plt.subplot(122)

animation = FuncAnimation(fig, animate_func, interval = 200, frames = np.arange(0, end_time * 1250, step * 1250, dtype = int))
plt.show()
