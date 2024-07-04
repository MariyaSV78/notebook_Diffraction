from __future__ import division
from math import *
import numpy as np
from pylab import *
from scipy import special

def Get_Chebyshev_Coefficients(Nc,Nc_max,amin,r):
	a=np.zeros(Nc_max,float)

	C=2
	#print('Nc initial=%d'%Nc)
	a[0] = special.jv(0,r)
	k = np.arange(1,Nc)
	a[k] = C*special.jv(k,r)

	while (a[Nc-1] > amin):
		a[Nc] = C*special.jv(Nc,r)
		Nc+=1

		if(Nc>Nc_max):
			print('The argument of the Chebyshev exp. is too large')
			break

	#print('r=%g'%r,', Nc=',Nc)

	return a[:Nc],Nc

def Get_Propagator_Chebyshev_Expansion(Psi,H,a,z):
#   This fucntion advances Psi in time units of delta t by 
#   the Chebyshev expansion of the propagator.
#   Function march corrected by A.Korovin

	Hz = H*z
	g0,g1 = Psi, dot(Hz,Psi)
	Psi = a[0]*g0 + a[1]*g1

	for jt in range(2,len(a)):
		g0,g1 = g1, g0 + dot(Hz,g1)*2 # reccurence of Chebyshev polynomials
		Psi += a[jt]*g1 # accumulation step

	Psi1 = dot(H,Psi)
	#Psi2 = dot(H,Psi1)

	#return Psi,Psi1,Psi2
	return Psi,Psi1

def Get_Propagator_Chebyshev_Expansion_old(N,Nc,Psi,H,a,z):
#   This fucntion advances Psi in time units of delta t by 
#   the Chebyshev expansion of the propagator.
#   Funtion march
	g0=np.zeros(N,complex)
	g1=np.zeros(N,complex)

	for i in range(N):
		g0[i] = Psi[i]
		Psi[i] = Psi[i]*a[0]    #g0(z)=1

#   get HPsi
	g1=dot(H,g0)    # HPsi

	for i in range(N):
		g1[i] = g1[i]*z
		Psi[i] =Psi[i]+g1[i]*a[1]   #g1(z)=z

	for jt in range(1,Nc-1):
		jp1 = jt+1

#       get HPsi
		g2=dot(H,g1)                #g2(z)== H*z

#       recurence relations of the chebichev expansion:
		for i in range(N):
#           normalization step
			g2[i] = g2[i]*z
#           reccurence of Chebyshev polynomials
			g2[i] = g2[i]*2.0 + g0[i]
			g0[i] = g1[i]
			g1[i] = g2[i]
#           accumulation step
			Psi[i] += a[jp1]*g1[i]

	Psi1=dot(H,Psi)
	Psi2=dot(H,Psi1)

	return Psi,Psi1,Psi2
