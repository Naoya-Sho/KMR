import numpy as np
import matplotlib.pyplot as plt
import random
from __future__ import division
from mc_tools import mc_compute_stationary, mc_sample_path
from discrete_rv import DiscreteRV

gamename = 'KMR'

pay = np.array([[[4, 4], [0, 3]], 
                [[3, 0], [2, 2]]]) 

N = 10
trials = 1000 
epsilon = 0.5
default = 0

def sep(a, p):
    return np.array([[p[0][0][a], p[0][1][a]],
                    [p[1][0][a], p[1][1][a]]])

pay_0 = sep(0, pay)


# Defining fuction to generate P
def genP(N, pay_0, epsilon):

    P = np.zeros([N+1, N+1]) 
    P[0, 0] = 1 - epsilon * 0.5
    P[0, 1] = epsilon * 0.5
    P[N, N] = 1 - epsilon * 0.5
    P[N, N-1] = epsilon * 0.5

    for n in range(1, N): 
        pr = (1 - n/N, n/N)
        expay = np.dot(pay_0, pr)

        if expay[0] == expay[1]:
            P[n, n+1] += (1 - n/N) * 0.5
            P[n, n-1] += n/N * 0.5

        elif expay[0] > expay[1]:
        
            P[n, n+1] += (1 - n/N) * epsilon * 0.5
            P[n, n-1] += n/N * (1 - epsilon * 0.5)

        else:
        
            P[n, n+1] += (1 - n/N) * (1 - epsilon * 0.5)
            P[n, n-1] += n/N * epsilon * 0.5


        P[n,n] = 1 - sum(P[n])

    return P


# Make a histogram of stationary dist

fig, ax = plt.subplots()

P = genP(N, pay_0, epsilon)
stat_dist = mc_compute_stationary(P)
ax.bar(range(N+1), stat_dist)
plt.title('Num. of players = ' + str(N)  + ', '+ 'Trials = '+ str(trials) + ', ' + 'epsilon = ' + str(epsilon), color='k')
plt.savefig(gamename + '_hist' + str(N) + str(trials) + str(epsilon) + '.png',transparent=True, bbox_inches='tight', pad_inches=0)
plt.show()

