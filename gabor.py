import matplotlib.pyplot as plt
import numpy as np

"""
builds Gabor filters
author: Niru Maheswaranathan
7:51 PM Nov 17, 2013
"""

# builds a 2D Gabor
def buildGabor(dim, sigma = 0.3, freq = 6):

    # create a range from -1 to 1 with the right number of points
    x = np.linspace(-1, -1, np.ceil(np.sqrt(dim)))

    # create a normal distribution
    y = (1/np.sqrt(2*np.pi*sigma))*np.exp(-.5*((x/sigma)**2))
    y = y/max(y)                      # normalize

    # get the outer product of y and y' (2D gaussian)
    filt = np.multiply.outer(np.transpose(y),y)

    # create sinusoid
    y2 = np.sin(x*np.pi*freq)

    # create grating
    y3  = np.ones(np.size(y2))
    img = np.multiply.outer(np.transpose(y3),y2)

    # mask gabor
    gabor = img*filt

    # return object with the right # of dimensions
    return gabor.flatten()[:dim]

if __name__=="__main__":

    print('hello')
    gabor = buildGabor(256)
