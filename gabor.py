import matplotlib.pyplot as plt
import numpy as np
import scipy

"""
builds Gabor filters
author: Niru Maheswaranathan
7:51 PM Nov 17, 2013
"""

# builds a 2D Gabor
def buildGabor(sigma = 0.3, freq = 6):

    x = scipy.r_[-1:1:.01]   #create a range from -1 to 1 stepped by .01

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

    # return gabor
    gabor = img*filt
    return gabor

if __name__=="__main__":

    print('hello')
    
    G = buildGabor()
    plt.imshow(G)
    plt.show()
