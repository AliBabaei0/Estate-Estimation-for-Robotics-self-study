#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 15:49:26 2021

@author: ali
"""

import pygame
from pygame.locals import *
import numpy as np
from numpy.linalg import inv
import sys

# Kalman Filter Implementation for discrete state
# step 1: Prediction
def predict(X_c_old, P_old, A, B, u, Q):
    ''' inputs:
         X_c_old:: corrected state vector from last update
         P_old:: corrected covariance matrix from last update
         u:: control signal
         Q:: process noise covariance matrix
         
        Outputs:
            X_p_new:: predictedd state vector 
            P_p_new:: projected covariance matrix
    '''
    X_p_new=A.dot(X_c_old)+B.dot(u);
    P_p_new=A.dot( (P_old.dot(A.T)) ) + Q;

    return X_p_new, P_p_new

# step 2: Update (Correction)
def update(X_p, P_p, H, z, R):
    '''
        Inputs:
            X_p:: predicted state
            P_p:: predicted covariance matrix
            H:: z=HX+v
            z:: measurement
            R:: measurement noise covariance matrix
        Outputs:
            X_c:: corrected state vector
            P_c:: corrected covariance matrix
            K:: Kalman Gain
    '''
    nom=P_p.dot(H.T)
    denom=inv( H.dot(nom)  + R)
    K = nom.dot(denom)
    X_c = X_p + K.dot( z- H.dot(X_p) ) 
    P_c = P_p - K.dot( H.dot(P_p) )
    return X_c, P_c , K

##############################################################################
#................... Initialize all of the Pygame modules ...................#
pygame.init()
window = pygame.display.set_mode((1024, 768), DOUBLEBUF)
screen = pygame.display.get_surface()
pygame.display.flip()
scale = 500
pygame.display.set_caption("Kalman Filter -  Simplified motion model - 2D Robot")

# initializaton
r = 0.1
dt = 1/8
ur = 2; ul = 2
rx= 0.05; ry= 0.075;
wx=0.10; wy=0.15;
A = np.eye(2)
B = dt*A
X = np.array([[0], [0]])
P = np.zeros((2,2))
H = np.array([[1,0],[0,2]])
Q=np.array([[wx**2, 0],[0, wy**2]]);
R=np.array([[ rx**2,0],[0,ry**2]]);
Xv=np.array([[0],[0]])

sf=120; # scale factor for displaying covariance error ellipse 
trace = [(0, 0), (0, 0)]
GT = [(0, 0), (0, 0)]
i = 0
screen.fill((0,0,0))
#image = pygame.image.load('drone.png')
while True:
    i += 1
    pygame.time.delay(100)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit();
    #screen.fill((255,255,255))
    #screen.blit(image, (100, 0))
    #pygame.transform.scale(screen, ((8,8)))
    
    if i<999:
        # predict
        W=np.array([[np.random.normal(0,wx**2)],[np.random.normal(0,wy**2)]]);
        u=np.array([[r/2*(ur+ul)],[r/2*(ur+ul)]])+W
        X, P = predict ( X, P, A, B, u, Q)
        
        rex = (X[0][0]*scale)
        rey = (X[1][0]*scale)
        Px = np.sqrt(P[0][0])*sf
        Py = np.sqrt(P[1][1])*sf
        trace[1] = (int(X[0][0]*scale), int(X[1][0]*scale))
        pygame.draw.lines(screen, (0, 0, 255), False, trace, 3)
        trace[0] = trace[1]
        
        GT[1] = (dt*i*r/2*(ur+ul)*scale, dt*i*r/2*(ur+ul)*scale)
        pygame.draw.lines(screen, (0, 250, 100), False, GT, 1)
        GT[0] = GT[1]
        
        
        if i%8==0:
            # update
            rxy=np.array([[np.random.normal(0,rx**2)],[np.random.normal(0,ry**2)]]);
            z = H.dot(X)+rxy
            X, P , K = update (X, P, H, z, R)
            Xv=np.append(Xv,X,1)
            
        #screen.blit(robot, (int(rex), int(rey)))
        pygame.draw.rect(screen, (255, 255, 1), (int(rex), int(rey), 5, 5))

        
        pygame.draw.ellipse(screen, (180, 40, 20), (rex-Px, rey-Py, 2*Px, 2*Py), 2)
            

    pygame.display.update()
    
    if X[0][0]>1.3:
        pygame.quit()


#plt.plot(Xv[0,:], Xv[1,:])
#plt.show()
