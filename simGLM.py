import numpy as np
from scipy.stats import poisson

"""
simulates a coupled GLM model
author: Niru Maheswaranathan
4:31 PM Nov 13, 2013
"""

def setParameters(n = 1, ds = 500, dh = 20, m = 50000, dt = 0.1):

    """

    Parameters
    ----------
    n : number of neurons
    ds: dimension of the stimulus
    dh: dimension of the coupling filters
    m : minibatch size (number of training examples)

    """

    # define the parameters dictionary
    params = {'n': n, 'ds': ds, 'dh': dh, 'm': m, 'dt': dt}
    return params

def generateModel(params):

    """

    Generates model parameters
    --------------------------

    Returns a theta dictionary:
    theta['w']:    stimulus filters for the n      neurons, (ds x n)        matrix
    theta['h']:    coupling filters for the n(n-1) neurons, (n  x dh x n-1) matrix

    """

    ## store model as dictionary
    theta = {}

    ## build the filters:
    # stimulus filters for each of n neurons
    theta['w'] = 0.2*np.random.randn(params['ds'], params['n'])

    # history (self-coupling) filters for each of n neurons
    theta['h'] = sort(0.1*np.rand(params['dh']+1, params['n']),0)
    theta['h'][-1] = 0                                              # current term is zero

    # coupling filters - stored as a big n by (n x dh) matrix
    #theta['h'] = np.zeros((params['dh']*params['n'], params['n']))

    #for idx in range(0,params['n']):

        ## generate coupling terms with other neurons
        #theta['h'][:,idx] = np.random.randn(params['dh']*params['n'])

        ## zero out self-coupling
        #theta['h'][np.arange(idx*params['dh'],(idx+1)*params['dh']),idx] = 0

    return theta


def f_df(theta, data, params):
    """
    objective assuming Poisson noise

    function [fval grad] = objPoissonGLM(theta, datapath)
    % Computes the Poisson log-likelihood objective and gradient
    % for the generalized linear model (GLM)
    """

    # fudge factor for numerical stability
    epsilon = 0

    M = data['x'].shape[0]

    # simulate the model at theta
    newData = simulate(theta, params, data['x'])
    rdt = newData['r']*params['dt']

    # compute objective value (negative log-likelihood)
    fval = sum(rdt - data['spkCount']*np.log(rdt + epsilon)) / M

    # compute gradient
    grad = dict()
    grad['w'] = (rdt - data['spkCount']).T.dot(data['x']).T / M

    return fval, grad



def simulate(theta, params, x = 'none', y = 'none'):

    """

    Simulates the output of the GLM
    -------------------------------
    """

    ## store output in dictionary
    data = {}

    # get stimulus
    if x == 'none':
        data['x'] = 0.2*np.random.randn(params['ds'], params['m']+params['dh']).T
    else:
        data['x'] = x

    # get coupling terms
    #if y == 'none':
        #y = np.random.randn(params['n']*params['dh'], params['m'])

    # compute stimulus response for the n neurons
    uw = theta['w'].T.dot(data['x'].T)
    stimResp = exp(uw)

    # compute history terms
    uh = convolve(squeeze(stimResp), squeeze(flipud(theta['h'])), 'valid')

    # compute coupling
    #coupResp = theta['h'].T.dot(y)

    # response of the n neurons (stored as an n by m matrix)
    data['r'] = np.exp(uw+uh).T

    return data

def genSpikes(rate):

    # draw samples from poisson distribution
    return poisson.rvs(rate)
