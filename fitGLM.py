import numpy as np
import simGLM as glm

"""
GLM maximum likelihood objectives
author: Niru Maheswaranathan
2:30 PM Nov 13, 2013
"""

def objPoissonGLM(theta, params, data):
    """
    objective assuming Poisson noise

    function [fval grad] = objPoissonGLM(theta, datapath)
    % Computes the Poisson log-likelihood objective and gradient
    % for the generalized linear model (GLM)
    """

    # fudge factor for numerical stability
    epsilon = 1e-12

    rdt = data['r']*params['dt']

    # compute objective value (negative log-likelihood)
    fval = data['spkCount'].dot((rdt - np.log(rdt + epsilon)).T)

    # compute gradient
    grad = (rdt - data['spkCount']).dot(data['x'].T)

    return (fval, grad)

if __name__=="__main__":

    print('Initializing parameters...')
    p = glm.setParameters()

    print('Generating model...')
    theta = generateModel(p)

    print('Simulating model...')
    data = simulate(theta, p)

    print('Drawing spike counts...')
    data['spkCount'] = genSpikes(data['r']*p['dt'])
