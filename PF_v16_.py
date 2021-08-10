import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform
from numpy.random import randn
import scipy
from numpy.linalg import norm
import scipy.stats
from filterpy.monte_carlo import systematic_resample


def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    #particles[:, 2] %= 2 * np.pi
    return particles

def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles

def predict(particles, u, ww, ws):
    """ MY PREDICTION"""

    N = len(particles)
    noise_angle=np.random.normal(0, ww,N)
    noise_dist=np.random.normal(0, ws,N)
    noise_dist2=np.random.normal(0, ws,N)
    # update heading
    Vr=u[0]
    Vl=u[1]
    omega=(Vr-Vl)/L
    V = (Vr+Vl)/2 
    particles[:, 2] += omega + noise_angle  #eq(6)
    # move in the (noisy) commanded direction
    err=np.ones(N)*ww**2
    particles[:, 0] += V * np.cos(particles[:, 2])  + noise_dist
    particles[:, 1] += V * np.sin(particles[:, 2])  + noise_dist2

def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        
        #distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        distance= np.linalg.norm(particles - landmark, axis=1)
        #metric=np.array([distance, particles[:,2]])
        weights *= scipy.stats.norm(distance,R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, np.random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)

def neff(weights):
    return 1. / np.sum(np.square(weights))

def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))


#from numpy.random import seed
#seed(2) 
#run_pf1(N=5000, plot_particles=False)
N=1000
iter=600

landmarks = np.array([[10, 10,0]])
NL = len(landmarks)
plt.figure()

# create particles and weights
particles = create_uniform_particles((0,20), (0,20), (0, 6.28), N)
weights = np.ones(N)*1/N

robot=np.array([15 ,10, 3.14/2])
#control signal
L=0.20 #Length of Robot 
u=np.array([0.5341/2,0.5131/2]) # it is Vr and Vl

for j in range (iter):
    
    # distance from robot to each landmark
    zs=np.sqrt( (robot[0]-landmarks[0,0])**2 + (robot[1]-landmarks[0,1])**2)+np.random.normal(0, 0.1)

    # GT
    Vr=u[0]
    Vl=u[1]
    omega=(Vr-Vl)/L
    robot[2] += omega
    V = (Vr+Vl)/2 
    robot[0] += V * np.cos(robot[2]) 
    robot[1] += V * np.sin(robot[2]) 

    
    # move particles to the next position acording to the control signal
    predict(particles, u, 0.1 , 0.01)
    
    # incorporate measurements
    z=np.array([zs])
    landmarks[0,2]=robot[2]
    if j%8==0:
        update(particles, weights, z, 0.1, landmarks)

    # resample if too few effective particles
    if neff(weights) < N/2:
        indexes = systematic_resample(weights)
        resample_from_index(particles, weights, indexes)
        assert np.allclose(weights, 1/N)
    mu, var = estimate(particles, weights)

    p0=plt.scatter(particles[:, 0], particles[:, 1],  color=(254/255,209/255,93/255), edgecolor='g', marker='.', s=500, alpha=0.33)
    p1 = plt.scatter(robot[0], robot[1], marker='x',color='k', s=75, lw=2)
    p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')
    p3=plt.scatter(landmarks[0,0],landmarks[0,1], marker='+', color=(0/255,128/255,128/255), s=100, lw=3)
    plt.legend([p0,p1, p2,p3], ['Particles','GT', 'Estimation', 'Landmark'], loc=2, numpoints=1)
    plt.xlim((3 ,17))
    plt.ylim((3 ,17))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.draw()
    plt.pause(0.00001)
    plt.clf()
    #robot[0]=mu[0]
    #robot[1]=mu[1]