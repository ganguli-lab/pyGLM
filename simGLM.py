import numpy as np
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

    # fudge factor for numerical stability
    epsilon = 0

    # number of data samples in this batch
    M = data['x'].shape[0]

    ## simulate the model at theta
    newData = simulate(theta, params, data['x'])
    rdt = newData['r']*params['dt']                 # rdt is: m by 1

    ## compute objective value (negative log-likelihood)
    fval = sum(rdt - data['spkCount']*np.log(rdt + epsilon)) / M

    ## compute gradient
    grad = dict()

    # temporarily store different in rate vs. observed spike count (used for gradient computations)
    rateDiff = (rdt - data['spkCount']).T           # rateDiff is: 1 by m

    # gradient for stimulus parameters
    grad['w'] = rateDiff.dot(data['x']).T / M       # grad['w'] is: ds by 1

    # gradient for history terms
    grad['h'] = np.zeros((params['dh']+1,1))

    # cross-correlate rate vector
    Cr = np.correlate(newData['stimResp'], np.squeeze(rateDiff), 'full')[params['dh']:rateDiff.size-1]
    grad['h'][:-1] = np.reshape( Cr[-params['dh']:] / M, (params['dh'],1) )
    #Cr = np.correlate(newData['rfull'],newData['rfull'],'full')[params['dh']:newData['rfull'].size-1]
    #grad['h'][:-1] = np.reshape( Cr[-params['dh']:] / M, (params['dh'],1) )
    #for t in np.arange(1,params['dh']+1):
        #grad['h'][-t-1] = params['dt']*rateDiff.dot(newData['rfull'][params['dh']-t:-t]) / M

    return fval, grad


def setParameters(ds = 500, dh = 10, m = 1000, dt = 0.1):

    """
    !! NOTE: Currently, n must be set to 1

    Parameters
    ----------
    n : number of neurons
    ds: dimension of the stimulus
    dh: dimension of the coupling filters
    m : minibatch size (number of training examples)

    """

    # define the parameters dictionary
    params = {'n': 1, 'ds': ds, 'dh': dh, 'm': m, 'dt': dt}
    return params

def generateModel(params):

    """

    Generates model parameters
    --------------------------

    Returns a theta dictionary:
    theta['w']:    stimulus filters for the n neurons, (ds x n) matrix
    theta['h']:    history  filters for the n neurons, (dh x n) matrix

    """

    ## store model as dictionary
    theta = {}

    ## build the filters:
    # stimulus filters for each of n neurons
    theta['w'] = 0.2*np.random.randn(params['ds'], params['n'])

    # history (self-coupling) filters for each of n neurons
    theta['h'] = np.sort(0.1*np.random.rand(params['dh']+1, params['n']),0)
    theta['h'][-1] = 0                                              # last term must be zero

    # coupling filters - stored as a big n by (n x dh) matrix
    #theta['h'] = np.zeros((params['dh']*params['n'], params['n']))

    #for idx in range(0,params['n']):

        ## generate coupling terms with other neurons
        #theta['h'][:,idx] = np.random.randn(params['dh']*params['n'])

        ## zero out self-coupling
        #theta['h'][np.arange(idx*params['dh'],(idx+1)*params['dh']),idx] = 0

    return theta


def simulate(theta, params, x = 'none', y = 'none'):

    """

    Simulates the output of the GLM
    -------------------------------
    """

    ## store output in dictionary
    data = {}

    # get stimulus (x is: m+dh by ds)
    if x == 'none':
        data['x'] = 0.2*np.random.randn(params['ds'], params['m']).T
    else:
        data['x'] = x

    # data size
    m = data['x'].shape[0] #- params['dh']

    if m < params['dh']:
        print('error: minibatch size is too small (smaller than history term)')

    # get coupling terms
    #if y == 'none':
        #y = np.random.randn(params['n']*params['dh'], params['m'])

    # compute stimulus response for the n neurons
    uw = theta['w'].T.dot(data['x'].T).T            # (uw is: m by 1)
    stimResp = np.squeeze( np.exp(uw) )
    #stimResp = np.exp(uw)

    # compute history terms                           (uh is: m by 1)
    uh_full = np.correlate(data['stimResp'], np.squeeze(theta['h']), 'full')
    uh = np.reshape(uh_full[:m],(m,1))

    # compute coupling
    #coupResp = theta['h'].T.dot(y)

    # response of the n neurons (stored as an n by m matrix)
    data['r'] = np.reshape( np.exp(uw+uh), (m,1) )

    # full response (including history buffer)
    #uw_full = np.vstack( (np.zeros((params['dh'],1)), uw ))
    #data['rfull'] = np.exp(np.squeeze(uw_full))# + uh_full)

    return data

def genSpikes(rate):

    # draw samples from poisson distribution
    return poisson.rvs(rate)


if __name__=="__main__":

    print('Initializing parameters...')
    p = setParameters(m = 1e4)

    print('Generating model...')
    theta = generateModel(p)

    print('Simulating model...')
    data = simulate(theta, p)

    print('Drawing spike counts...')
    data['spkCount'] = genSpikes(data['r']*p['dt'])

    print('Evaluating objective...')
    fval, grad = f_df(theta, data, p)
