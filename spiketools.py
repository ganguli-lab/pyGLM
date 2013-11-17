import scipy.signal as sps
import numpy as np

"""
A set of spike train tools
author: Niru Maheswaranathan (nirum@stanford.edu)
date:   Sept. 16th, 2013
"""

# smooth a set of spike trains with a gaussian filter
def smoothspikes(t, s, sigma):

    # time step
    dt = np.mean(np.diff(t))

    # make the gaussian kernel
    tau = np.arange(-5*sigma,5*sigma,dt)
    kernel = dt*normpdf(tau, 0, sigma)

    # smooth it
    return correlate(s, kernel, 'same')

# extract spikes within a certain ISI band
def isiband(spikeTimes, band):

    # compute ISIs
    spkdiff = np.concatenate([ np.array([spikeTimes[0]]), np.diff(spikeTimes) ])

    # compute valid spikes
    valid = (spkdiff > band[0]) & (spkdiff <= band[1])

    # return spikes in this band
    return spikeTimes[valid]

# bin spikes
def binspikes(spikes, time):

    spk = np.zeros(np.shape(time))
    dt = np.mean(np.diff(time))

    for j in np.arange(np.size(time)):
        spk[j] = sum((spikes > time[j]-dt) & (spikes <= time[j]))

    return spk
