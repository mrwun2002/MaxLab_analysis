import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
#from read_raw import load_from_file
import maxlab_analysis as mla

import time
from scipy.signal import find_peaks
import scipy.stats as stats
from sklearn.decomposition import PCA



filename = "div28.data.raw.h5"



data_from_npy = np.load(filename + '.npy', mmap_mode = 'r', )
plt.plot(data_from_npy[:, 0], data_from_npy[:, 1:50])

n_components = 3

num_points = 500000

t = data_from_npy[0:num_points, 0]
X = data_from_npy[0:num_points, 1:863:5]
pca = PCA(n_components)
pca.fit(X)

print(pca.explained_variance_ratio_)
print(len(pca.explained_variance_ratio_))
print(pca.components_)
X_new = pca.transform(X)
print(X_new)



fig = plt.figure()
ax = fig.add_subplot(projection='3d')
p = ax.scatter(X_new[:, 0], X_new[:, 1], X_new[:, 2], s = 1, c = t)
fig.colorbar(p)
fig.show()

# Rotate the axes and update
for angle in range(0, 360*4 + 1):
    # Normalize the angle to the range [-180, 180] for display
    angle_norm = (angle + 180) % 360 - 180

    # Cycle through a full rotation of elevation, then azimuth, roll, and all
    elev = azim = roll = 0
    if angle <= 360:
        elev = angle_norm
    elif angle <= 360*2:
        azim = angle_norm
    elif angle <= 360*3:
        roll = angle_norm
    else:
        elev = azim = roll = angle_norm

    # Update the axis view and title
    ax.view_init(elev, azim, roll)
    plt.title('Elevation: %d°, Azimuth: %d°, Roll: %d°' % (elev, azim, roll))

    plt.draw()
    plt.pause(.001)