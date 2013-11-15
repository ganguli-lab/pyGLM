import numpy as np
import simGLM as glm

"""
GLM maximum likelihood objectives
author: Niru Maheswaranathan
2:30 PM Nov 13, 2013
"""

if __name__=="__main__":

    print('Initializing parameters...')
    p = glm.setParameters()

    print('Generating model...')
    theta = generateModel(p)

    print('Simulating model...')
    data = simulate(theta, p)

    print('Drawing spike counts...')
    data['spkCount'] = genSpikes(data['r']*p['dt'])
