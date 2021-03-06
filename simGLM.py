import numpy as np
import pyGLM.gabor as gabor
from scipy.stats import poisson
from os.path import join

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
    rhat, expu = simulate(theta, params, data)
    rdt  = rhat*params['dt']                        # rdt is: M by N

    ## compute objective value (negative log-likelihood)
    fval = np.sum(rdt - data['n']*np.log(rdt + epsilon)) / M

    ## compute gradient
    grad = dict()

    # temporarily store different in rate vs. observed spike count (used for gradient computations)
    #rateDiff = (rdt - data['n']).T                  # rateDiff is: N by M  (used for exponential nonlinearity)
    rateDiff = ((params['dt'] - data['n']/rhat)*expu).T                  # rateDiff is: N by M  (used for soft linear rectifying nonlinearity)

    # gradient for stimulus parameters
    grad['w'] = rateDiff.dot(data['x']).T / M       # grad['w'] is: ds by N

    # gradient for the offset term
    grad['b'] = np.sum(rateDiff.T, axis=0) / M                 # grad['b'] is:  1 by N

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
            grad['h'][dh-delta,nrnIdx] = np.sum( spkCountArray[:-delta] * rateDiffArray[delta:] , axis=0) / M

    # check for nans
    gradCheck(grad)

    return fval, grad

def setParameters(n = 10, ds = 1024, dh = 50, m = 1000, dt = 0.1):

    """
    Parameters
    ----------
    n : number of neurons
    ds: dimension of the stimulus
    dh: dimension of the coupling filters
    m : minibatch size (number of training examples)

    """

    # define the parameters dictionary
    params = {'numNeurons': n, 'stim_dim': ds, 'hist_dim': dh, 'numSamples': m, 'dt': dt, 'alpha': 0.1}

    # nonlinearity
    params['f'] = logexp # (soft-linear-rectifying)
    #params['f'] = np.exp # exponential

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
            mySigma = np.random.rand()*0.6 + 0.1
            myFreq  = np.random.rand()*5
            myPhase = np.random.rand()*2*np.pi

            # build filter
            theta['w'][:,nrnIdx] = gabor.buildGabor(ds, sigma=mySigma, freq=myFreq, phase=myPhase)

    else:
        print('WARNING: unrecognized filter type. Using random values instead.')
        theta['w'] = 0.2*np.random.randn(ds, N)

    # pick out two pools
    fracInh = 0.2
    numInh = np.ceil( fracInh * N )
    numExc = N - numInh

    # normalize filters
    #theta['w'] = theta['w'] / np.linalg.norm( theta['w'], axis=0 )
    theta['w'] = 1.5*theta['w'] / np.sqrt(np.sum(theta['w']**2, axis=0))

    # offset (scalar)
    #theta['b'] = -1*np.ones((1,N)) + 0.25*np.random.randn(1,N)
    theta['b'] = np.zeros((1,N))

    theta['b'][0,:numInh] = 0.15*np.ones((1,numInh))
    theta['b'][0,numInh:] = -5 + 3*np.random.rand(1,numExc)

    # history (self-coupling) filters for each of n neurons
    theta['h'] = np.zeros( (dh,N) )

    # random decay (linear)
    #theta['h'][:,:numInh] = -0.2*np.sort( np.random.rand( dh, numInh ), axis=0 )
    #theta['h'][:,numInh:] = -0.5*np.sort( np.random.rand( dh, numExc ), axis=0 )

    # exponential decay
    tau = np.linspace(1,0,dh).reshape( (dh,1) )
    theta['h'][:,:numInh] = -0.2*np.exp(-10*tau.dot( np.ones( (1, numInh) )))
    theta['h'][:,numInh:] = -0.5*np.exp(-5*tau.dot(  np.ones( (1, numExc) )))

    # inhibitory connections
    inhK = -5.5*np.random.rand(N, numInh) / numInh

    # excitatory (sparse) connections
    excK = 10*np.random.rand(N, numExc) / (numExc * params['alpha'])
    temp =    np.random.rand(N, numExc)
    excK[temp >= params['alpha']] = 0

    # coupling filters - stored as a (n by n) matrix
    theta['k'] = np.hstack(( inhK, excK )).T
    theta['k'] -= np.diag(np.diag(theta['k']))

    return theta

def logexp(u):
    #for stability
    epsilon = 1e-6

    output = np.log( 1 + np.exp( u )) + epsilon
    output[u > 50] = u[u > 50]
    return output

def loadExternalData(stimFile, ratesFile, shapes, baseDir='.'):

    data = dict()

    # load stimuli
    data['x'] = np.memmap(join(baseDir, stimFile), dtype='uint8', mode='r', shape=tuple(shapes['stimSlicedShape']))

    # load rates
    data['n'] = np.memmap(join(baseDir, ratesFile), dtype='uint8', mode='r', shape=tuple(shapes['rateShape']))[40:,:]

    return data

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

    # nonlinearity
    f = params['f']

    # model parameters
    w = theta['w']
    b = theta['b']
    h = theta['h']
    k = theta['k']

    # offset for numerical stability
    epsilon = 1e-20

    ## store output in a dictionary
    data = {}

    # output
    data['n'] = np.zeros( (M, N) )                   # spike history
    data['r'] = np.zeros( (M, N) )                   # spike history

    # generate stimuli
    #data['x'] = 0.2*np.random.randn(M, ds)         # Gaussian white noise
    data['x'] = genPinkNoise(M, np.sqrt(ds), params['dt']).reshape( (M, ds) )

    # compute stimulus projection
    #u = w.T.dot(data['x'].T).T            # (u is: M by N)
    u = data['x'].dot(w)

    # the initial rate (no history)
    data['n'][0,:] = poisson.rvs( f( u[0,:] + b ) )
    data['r'][0,:] = f( u[0,:] + b )

    # the next rate (one history point)
    data['n'][1,:] = poisson.rvs( f( u[1,:] + data['n'][0,:]*h[-1,:] + b ) )
    data['r'][1,:] = f( u[1,:] + data['n'][0,:]*h[-1,:] + b )

    # simulate the model (in time)
    for j in np.arange(2,M):

        # store output
        v = np.zeros((1,N))

        # compute history weights
        if j < dh+1:
            v = np.sum( data['n'][:j,:]*h[-j:] , axis=0)
            # for each neuron
            #for nrnIdx in range(N):
                #n1 = np.squeeze(data['n'][0:j,nrnIdx])
                #n2 = np.squeeze(h[:,nrnIdx])
                #v[0,nrnIdx] = np.correlate( n1 , n2 , 'valid' )[0]

        else:
            # for each neuron
            v = np.sum(data['n'][j-dh:j,:]*(h), axis=0)

        # print out contributions
        #print('stim: %g\thistory: %g\tcoupling: %g\tbias: %g'%(np.linalg.norm(u[j,:]), np.linalg.norm(v), np.linalg.norm(data['n'][j-1,:].dot(k)), np.linalg.norm(b)))

        # compute model firing rate
        r = f( u[j,:] + v + b + data['n'][j-1,:].dot(k) ) + epsilon
        data['r'][j,:] = r.copy()

        # cap spike count
        maxVal = 10
        r[r > maxVal] = maxVal

        # draw spikes
        spikes = poisson.rvs(r)

        # store
        data['n'][j,:] = spikes

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

    # nonlinearity
    f = params['f']

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
    #u = w.T.dot(x.T).T
    u = x.dot(w)

    # compute coupling projection
    kappa = np.vstack(( np.zeros((1,N)), n.dot(k)[1:,:] )) # (kappa is: M by N)

    # predicted rates
    rates = np.zeros( (M,N) )
    expu  = np.zeros( (M,N) )

    # compute history terms                             # (uh is: M by N)
    for nrnIdx in range(N):
        v = np.reshape( np.correlate(n[:,nrnIdx], h[:,nrnIdx], 'full')[0:n.shape[0]-1], (M-1,1) )
        v = np.vstack(( np.zeros((1,1)) , v ))

        # input to the nonlinearity
        linearOutput = u[:,nrnIdx] + np.squeeze(v) + b[0,nrnIdx] + kappa[:,nrnIdx]
        arrayCheck(linearOutput, 'linear output')

        # store exp(linearOtput) for the gradient
        expu[:,nrnIdx]  = np.exp(linearOutput) / ( 1 + np.exp(linearOutput) )
        expu[linearOutput > 50, nrnIdx] = 1

        arrayCheck(expu[:,nrnIdx], 'exp(u) / (1 + exp(u)) for neuron %g'%(nrnIdx))

        # response of the n neurons (stored as an n by m matrix)
        rates[:,nrnIdx] = f( linearOutput )
        arrayCheck(rates[:,nrnIdx], 'rhat for neuron %g'%(nrnIdx))

    return rates, expu

def visualizeNetwork(theta, params, data):

    print('\n====================================')
    print('===== Simulated GLM Properties =====')

    print('\nSpike Counts:')
    print('mean: ', np.mean(data['n']))
    print('var.: ', np.var(data['n']))

    print('\nRates:')
    print('mean: ', np.mean(data['r']))
    print('var.: ', np.var(data['r']))
    print('*** %g percent of rates are over the limit. ***'%(100*np.mean(data['r']>10)))

    print('\n====================================\n')

def genPinkNoise(t, n, dt):

    noise = np.random.randn(t,n,n)

    tc = np.arange(t).reshape(( -1,1,1 )) * dt
    xc = np.arange(n).reshape(( 1,-1,1 ))
    yc = np.arange(n).reshape(( 1,1,-1 ))

    tc -= np.mean(tc)
    xc -= np.mean(xc)
    yc -= np.mean(yc)

    radii = np.sqrt( tc**2 + xc**2 + yc**2 )

    # normalize
    noise /= np.fft.fftshift(radii)

    # generate via ifft
    stim = np.real(np.fft.ifftn(noise))
    stim /= np.sqrt(np.mean(stim**2))

    return stim

def gradCheck(grad):
    for key in grad.keys():
        arrayCheck(grad[key], key)

def arrayCheck(arr, name):
    if ~np.isfinite(arr).all():
        print('**** WARNING: Found non-finite value in ' + name + ' (%g percent of the values were bad)'%(np.mean(np.isfinite(arr))))




if __name__=="__main__":

    print('Initializing parameters...')
    p = setParameters(n = 100, ds = 256, dh = 10, m = 1e4)

    print('Generating model...')
    theta = generateModel(p, 'gabor')

    print('Simulating model...')
    data = generateData(theta, p)
    visualizeNetwork(theta, p, data)

    from matplotlib.pylab import *

    # plot connections
    figure(7)
    clf()
    imshow(theta['k'])
    colorbar()
    draw()

    # plot rasters
    figure(6)
    clf()
    imshow(np.log(1 + data['n'].T))
    colorbar()
    draw()

    figure(8)
    clf()
    imshow(np.log(data['r'].T))
    colorbar()
    draw()

    show()

    print('Evaluating objective...')
    fval, grad = f_df(theta, data, p)

    print('Objective: %g'%(fval))

    for key in grad.keys():
        print('grad[' + key + ']: %g'%(np.linalg.norm(grad[key])))
