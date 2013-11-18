import numpy as np
import gabor
from scipy.stats import poisson

"""
simulates a coupled GLM model
author: Niru Maheswaranathan
4:31 PM Nov 13, 2013
"""

def f_df(theta, data, params):
    """
    log-likelihood objective and gradient, assuming Poisson noise

    function [fval grad] = f_df(theta, data, params)
    % Computes the Poisson log-likelihood objective and gradient
    % for the generalized linear model (GLM)
    """

    ## get out parameters
    ds = params['stim_dim']
    dh = params['hist_dim']
    N  = params['numNeurons']
    M = data['x'].shape[0]

    # offset for numerical stability
    epsilon = 1e-20

    ## simulate the model at theta
    rhat = simulate(theta, params, data)
    rdt  = rhat*params['dt']                        # rdt is: M by N

    ## compute objective value (negative log-likelihood)
    fval = sum(sum(rdt - data['n']*np.log(rdt + epsilon))) / M

    ## compute gradient
    grad = dict()

    # temporarily store different in rate vs. observed spike count (used for gradient computations)
    rateDiff = (rdt - data['n']).T                  # rateDiff is: N by M

    # gradient for stimulus parameters
    grad['w'] = rateDiff.dot(data['x']).T / M       # grad['w'] is: ds by N

    # gradient for the offset term
    grad['b'] = sum(rateDiff.T) / M                 # grad['b'] is:  1 by N

    # gradient for history terms
    grad['h'] = np.zeros((dh,N))                    # grad['h'] is: dh by N

    # gradient for coupling terms
    nprev = np.vstack(( np.zeros((1,N)), data['n'][1:,:] )) # (nprev is: M by N)
    grad['k'] = rateDiff.dot(nprev).T / M           # grad['k'] is:  N by N

    # for each neuron
    for nrnIdx in range(N):

        # cross-correlate rate vector
        spkCountArray = np.squeeze(data['n'][:,nrnIdx])
        rateDiffArray = np.squeeze(rateDiff[nrnIdx,:])

        # at each time offset
        for delta in np.arange(1,dh+1):

            # average product of spike counts and rateDiff at each point in time
            grad['h'][dh-delta,nrnIdx] = sum( spkCountArray[:-delta] * rateDiffArray[delta:] ) / M

    return fval, grad

def setParameters(n = 20, ds = 500, dh = 10, m = 1000, dt = 0.1):

    """
    Parameters
    ----------
    n : number of neurons
    ds: dimension of the stimulus
    dh: dimension of the coupling filters
    m : minibatch size (number of training examples)

    """

    # define the parameters dictionary
    params = {'numNeurons': n, 'stim_dim': ds, 'hist_dim': dh, 'numSamples': m, 'dt': dt}
    return params

def generateModel(params, filterType='gabor'):

    """

    Generates model parameters
    --------------------------

    Returns a theta dictionary:
    theta['w']:    stimulus filters for the n neurons, (ds x N) matrix
    theta['b']:    stimulus offset  for the n neurons, ( 1 x N) matrix
    theta['h']:    history  filters for the n neurons, (dh x N) matrix
    theta['k']:    coupling filters for the n neurons, ( N x N) matrix

    """

    ## get out parameters
    ds = params['stim_dim']
    dh = params['hist_dim']
    N  = params['numNeurons']
    M  = params['numSamples']

    ## store model as dictionary
    theta = {}

    ## build the filters:

    # stimulus filters for each of n neurons
    if filterType is 'random':
        theta['w'] = np.random.randn(ds, N)

    elif filterType is 'sinusoid':
        theta['w'] = np.zeros((ds, N))
        for nrnIdx in range(N):
            theta['w'][:,nrnIdx] = np.sin( np.linspace(0,2*np.pi,ds) + 2*np.pi*np.random.rand() )

    elif filterType is 'gabor':
        theta['w'] = np.zeros((ds, N))
        for nrnIdx in range(N):

            # random parameters
            mySigma = np.random.rand()*5 + 0.1
            myFreq  = np.random.rand()*10

            # build filter
            theta['w'][:,nrnIdx] = gabor.buildGabor(ds, sigma=mySigma, freq=myFreq)

    else:
        print('WARNING: unrecognized filter type. Using random values instead.')
        theta['w'] = 0.2*np.random.randn(ds, N)

    # normalize filters
    #theta['w'] = theta['w'] / np.linalg.norm( theta['w'], axis=0 )
    theta['w'] = theta['w'] / np.sqrt(np.sum(theta['w']**2, axis=0))

    # offset (scalar)
    theta['b'] = -1*np.ones((1,N)) + 0.1*np.random.randn(1,N)

    # history (self-coupling) filters for each of n neurons
    theta['h'] = -0.1*np.sort( np.random.rand( dh, N ), axis=0 )

    # coupling filters - stored as a (n by n) matrix
    theta['k'] = np.random.rand(N, N)
    theta['k'] = (theta['k'] - theta['k'].T)/2

    return theta

def generateData(theta, params):

    """
    Generates stimuli and draws spike counts from the model
    -------------------------------------------------------
    """

    ## get out parameters

    # constants
    ds = params['stim_dim']
    dh = params['hist_dim']
    N  = params['numNeurons']
    M  = params['numSamples']

    # model parameters
    w = theta['w']
    b = theta['b']
    h = theta['h']
    k = theta['k']

    # offset for numerical stability
    epsilon = 1e-20

    ## store output in a dictionary
    data = {}

    # input / output
    data['x'] = 0.2*np.random.randn(M, ds) # stimulus
    data['n'] = np.zeros( (M, N) )                   # spike history

    # compute stimulus projection
    u = w.T.dot(data['x'].T).T            # (u is: M by N)

    # the initial rate (no history)
    data['n'][0,:] = poisson.rvs( np.exp( u[0,:] + b ) )

    # the next rate (one history point)
    data['n'][1,:] = poisson.rvs( np.exp( u[1,:] + data['n'][0,:]*h[-1,:] + b ) )

    # simulate the model (in time)
    for j in np.arange(2,M):

        # store output
        v = np.zeros((1,N))

        # compute history weights
        if j < dh+1:

            # for each neuron
            for nrnIdx in range(N):
                n1 = np.squeeze(data['n'][0:j,nrnIdx])
                n2 = np.squeeze(h[:,nrnIdx])
                v[0,nrnIdx] = np.correlate( n1 , n2 , 'valid' )[0]

        else:

            # for each neuron
            v = sum(data['n'][j-dh:j,:]*(h))

        # compute model firing rate
        r = np.exp( u[j,:] + v + b + data['n'][j-1,:].dot(k) ) + epsilon

        # draw spikes
        data['n'][j,:] = poisson.rvs(r)

    return data

def simulate(theta, params, data):

    """
    Simulates the output of the GLM given true stimulus/response
    ------------------------------------------------------------
    """

    ## get out parameters

    # constants
    ds = params['stim_dim']
    dh = params['hist_dim']
    N  = params['numNeurons']
    M = data['x'].shape[0]

    # model parameters
    w = theta['w']
    b = theta['b']
    h = theta['h']
    k = theta['k']

    # get stimuli, offset, and spike counts
    x = data['x']           # stimuli       (x is: M by ds)
    n = data['n']           # spike counts  (n is: M by N)

    # make sure we have a reasonable number of samples
    if M < dh:
        print('Error: minibatch size is too small (smaller than history term)')

    # compute stimulus projection for the n neurons     # (u is: M by N)
    u = w.T.dot(x.T).T

    # compute coupling projection
    kappa = np.vstack(( np.zeros((1,N)), n.dot(k)[1:,:] )) # (kappa is: M by N)

    # predicted rates
    rates = np.zeros( (M,N) )

    # compute history terms                             # (uh is: M by N)
    for nrnIdx in range(N):
        v = np.reshape( np.correlate(np.squeeze(n[:,nrnIdx]), np.squeeze(h[:,nrnIdx]), 'full')[0:n.shape[0]-1], (M-1,1) )
        v = np.vstack(( np.zeros((1,1)) , v ))

        # response of the n neurons (stored as an n by m matrix)
        rates[:,nrnIdx] = np.exp( u[:,nrnIdx] + np.squeeze(v) + b[0,nrnIdx] + kappa[:,nrnIdx])

    return rates

if __name__=="__main__":

    print('Initializing parameters...')
    p = setParameters(n = 2, ds = 50, dh = 10, m = 1e4)

    print('Generating model...')
    theta = generateModel(p, 'gabor')

    print('Simulating model...')
    data = generateData(theta, p)

    print('Evaluating objective...')
    fval, grad = f_df(theta, data, p)
