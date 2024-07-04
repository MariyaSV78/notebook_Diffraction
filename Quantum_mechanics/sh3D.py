#******************************************************************************#
#    December 2017
#    Author: MEHDI AYOUZ LGPM CENTRALESUPELEC
#    mehdi.ayouz@centralesupelec.fr
#    Co-authors: V. Kokoouline, JM Gillet and PE Janolin
#    input: N, x0, xL, m, V
#    output : E[0:Nx*2Nphi-1] (eigen energies)
#    psi[0:Nx-1,0:2Nphi+1] (normalized eigen fonctions)    
#******************************************************************************#
from __future__ import division
from math import *
import numpy as np
#from numpy import linalg as LA
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import scipy.special
import numpy.polynomial.legendre

from .constants import ci


#*****************************************************************************80
## P_POLYNOMIAL_PRIME evaluates the derivative of Legendre polynomials P(n,x).
#
#  Discussion:
#
#    P(0,X) = 1
#    P(1,X) = X
#    P(N,X) = ( (2*N-1)*X*P(N-1,X)-(N-1)*P(N-2,X) ) / N
#
#    P'(0,X) = 0
#    P'(1,X) = 1
#    P'(N,X) = ( (2*N-1)*(P(N-1,X)+X*P'(N-1,X)-(N-1)*P'(N-2,X) ) / N
#
#  Parameters:
#
#    Input, integer N, the highest order polynomial to evaluate.
#    Note that polynomials 0 through N will be evaluated.
#
#    Input, real x, the evaluation points.
#
#    Output, real dPl(N), the values of the derivatives of the
#    Legendre polynomials of order 0 through N-1 at the point X.
#    also Pl as  the Legendre polynomials of order 0 through N-1
#     at the point X
#*****************************************************************************#
def Get_Grid3D(Nx,Nphi): 
	x=np.zeros(Nx,float)
	w=np.zeros(Nx,float)
	delta_x= np.zeros(Nx,float)

	dPNx=np.zeros(Nx,float)

#   Build x root of the Legendre polynomial
	x,w=numpy.polynomial.legendre.leggauss(Nx)

#	phi= np.zeros(2*Nphi+2,float)     # declare of the grid point along phi
#	for j in range(2*Nphi+1):        # build the phi grid point         
#		phi[j+1]=(j+1)*2*pi/(2*Nphi+1)
#	phi[0]=0.

#	theta= np.zeros(Nx,float)     # declare of the grid point along theta
#	for i in range(Nx):        # build the theta grid point
#		theta[i]=acos(x[Nx-1-i])

	theta = np.linspace(0,pi,Nx)
	phi = np.linspace(0,2*pi,Nx)
	delta_phi = phi[1]-phi[0]

	for i in range(Nx):
		dPNx[i]=Get_Legendre_Polynomial_Derivative(Nx,x[i])
		delta_x[i]=(2.)/(1-x[i]**2)/dPNx[i]**2

	return x,w,theta,phi,delta_x,delta_phi

#*****************************************************************************#
def Get_Analytical_Solutions3D(Nx,Nphi,l,m,theta,phi,psi=0):
#   Analytic solution using spherical harmonics 
#	Ylm=np.zeros((Nx,2*Nphi+2),float)
	Ylm=np.zeros((Nx,Nx),float)
	Lsquare_analytic=np.zeros(Nx*(2*Nphi+1),float)
	Lmodulus_analytic=np.zeros(Nx*(2*Nphi+1),int)
	Lz_analytic=np.zeros(Nx*(2*Nphi+1),int)

	deviation=0.
	tmp_Ylm=complex(0.,0.)

#    for i in range(N):
#        for j in range(N):
#            Ylm[i,j]=scipy.special.sph_harm(m,l,phi[i],theta[j]).real  

#	for j in range(2*Nphi+2):
	for j in range(Nx):
		for i in range(Nx):
			tmp_Ylm=scipy.special.sph_harm(m,l,theta[i],phi[j])
			Ylm[i,j]=abs(tmp_Ylm)
			#Ylm[i,j]=abs(-tmp_Ylm+scipy.special.sph_harm(-m,l,phi[j],theta[i]))/sqrt(2.)
			# for Px orbital, for example
			#Ylm[i,j]=abs(ci*tmp_Ylm+ci*scipy.special.sph_harm(-m,l,phi[j],theta[i]))/sqrt(2.)
			# for Py orbital, for example
			#Ylm[i,j]=abs(tmp_Ylm+ci*scipy.special.sph_harm(-m,l,phi[j],theta[i]))/sqrt(2.)
			# for f_z(x^2-y^2) orbital, for example

			if not psi==0:
				deviation += abs(abs(psi[i,j])-Ylm[i,j])/Ylm[i,j]
	print('Relative deviation between psi and Ylm =%g '% (deviation*100/(Nx*(2*Nphi+2))))

#   Build analytic eigen values
	i=0
	for ll in range(Nx*(2*Nphi+1)):
		ml=0
		while (ml<2*ll+1 and i< Nx*(2*Nphi+1)):
			Lsquare_analytic[i]=ll*(ll+1)
			Lmodulus_analytic[i]=ll
			Lz_analytic[i]=ml-ll
			i=i+1
			ml=ml+1
			
	return Ylm,Lsquare_analytic,Lmodulus_analytic,Lz_analytic

	THETA, PHI=meshgrid(theta,phi)    
	R=transpose(Ylm)

	fig = plt.figure(2)
	ax = fig.gca(projection='3d')
	
	X = R * sin(THETA) * cos(PHI)
	Y = R * sin(THETA) * sin(PHI)
	Z = R * cos(THETA)   
	surf = ax.plot_surface(X, Y, Z,rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
	ax.set_xlabel('$x$',size=20)
	ax.set_ylabel('$y$',size=20)
	ax.set_zlabel('$z$',size=20)
	title("$|\mathcal{Y}_{l=0}^{m=0}(\\theta,\phi)|$",size=20)
	#ax.set_zlim3d(-1, 1)
	#fig.colorbar(surf)
	plt.show()








#*****************************************************************************#
def Get_Legendre_Polynomial_Derivative(N,x):
	
	if(N==0):
		PN=1e0;dPN=0e0
		
	else:  
		PN=x; PNm1=1e0
		
		for i in range(2,N+1):
			PNm2=PNm1; PNm1=PN
			PN=((2*i-1)*x*PNm1-(i-1)*PNm2)/i

		dPN=N*(x*PN-PNm1)/(x**2-1e0) if(x**2-1e0) else 0.5*N*(N+1)*PN/x
		
	return dPN

#*****************************************************************************#        
def Get_Kinetic_Operator(Nx,Nphi,x):
	Lsquare_x=np.zeros((Nx,Nx),float)
	Lsquare_phi=np.zeros((2*Nphi+1,2*Nphi+1),float)

	for i in range(Nx):
		for ip in range(Nx):
			if(i==ip):
				Lsquare_x[i,ip]=1/3.*(Nx*(Nx+1)-2./(1-x[i]**2))

			elif(i!=ip):
				Lsquare_x[i,ip]=2*sqrt(1-x[i]**2)*sqrt(1-x[ip]**2)/(x[i]-x[ip])**2

	for j in range(2*Nphi+1): 
		for jp in range(2*Nphi+1):
			if(j==jp):
				Lsquare_phi[j,jp]=Nphi*(Nphi+1)/3

			elif(j!=jp):
				Lsquare_phi[j,jp]=(-1)**(j-jp)*cos(pi*(j-jp)/(2*Nphi+1))/(2*sin(pi*(j-jp)/(2*Nphi+1))**2)    

			
	return Lsquare_x,Lsquare_phi

#*****************************************************************************#
def Get_Angular_Momentum_Opertor(Nx,x,Nphi,Lsquare_x,Lsquare_phi):
	Lsquare=np.zeros((Nx*(2*Nphi+1),Nx*(2*Nphi+1)),float) 

	for i in range(Nx):
		for ip in range(Nx):
			for j in range(2*Nphi+1):
				for jp in range(2*Nphi+1):
					k=i+j*Nx
					l=ip+jp*Nx

					Lsquare[k,l]=Lsquare_x[i,ip]*delta(j,jp)+Lsquare_phi[j,jp]*delta(i,ip)/(1-x[i]**2)

	return Lsquare
#*****************************************************************************# 
def delta(i,j):
	delta=0
	if(i==j):
		delta=1
	return delta
#*****************************************************************************#    
def Get_Wave_Functions_Normalization(Nx,Nphi,n,delta_x,delta_phi,P,psi):
	psi=np.zeros((Nx,2*Nphi+2),float)

#   Save the wave function of the nth state
	for i in range(Nx):
		for j in range(2*Nphi+1):
			psi[i,j+1]=P[i+Nx*j,n]

#   Add the symmetrical point psi(theta,0)=psi(theta,2pi)
	psi[:,0]=psi[:,2*Nphi+1]

	norm=0
	for i in range(Nx):
		for j in range(2*Nphi+2):
			norm=norm+psi[i,j]**2

	psi=psi/sqrt(abs(norm))
	
#   DVR transformation
	for i in range(Nx):
		for j in range(2*Nphi+2):
			psi[i,j]=psi[i,j]/(sqrt(delta_x[i])*sqrt(delta_phi))

#   Checkout the normalization
	norm=0.
	for i in range(Nx):
		for j in range(2*Nphi+2):
			norm=norm+psi[i,j]**2*delta_x[i]*delta_phi  

	print('Checkout norm=%g for n=%d'%(norm,n))  

	return psi

#*****************************************************************************#
def Get_Wave_Functions_Plot(Nx,Nphi,theta,phi,delta_x,delta_phi,psi):

	THETA, PHI=meshgrid(theta,phi) 
	R=transpose(abs(psi))
 
	fig = plt.figure(1)
	ax = fig.gca(projection='3d')
	
	X = R * sin(THETA) * cos(PHI)
	Y = R * sin(THETA) * sin(PHI)
	Z = R * cos(THETA)   
	surf = ax.plot_surface(X, Y, Z,rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
	ax.set_xlabel('$x$',size=20)
	ax.set_ylabel('$y$',size=20)
	ax.set_zlabel('$z$',size=20)
	title("$|\psi_{n=13}(\\theta,\phi)| \\rightarrow 1s $",size=20)
	#ax.set_zlim3d(-1, 1)
	#fig.colorbar(surf)
	plt.show()
	
	return None
#*****************************************************************************#
def Get_Hybrisation(Nx,Nphi,theta,phi,delta_x,delta_phi, n,P):

	hybridation=np.zeros((Nx,2*Nphi+2),float)  
#   Save the wave function of the nth state
	for i in range(Nx):
		for j in range(2*Nphi+1):
			hybridation[i,j+1]=(P[i+Nx*j,0]+P[i+Nx*j,3])/sqrt(2)

#   Add the symmetrical point hybridation(theta,0)=hybridation(theta,2pi)
	hybridation[:,0]=hybridation[:,2*Nphi+1]

	norm=0.        
	for i in range(Nx):
		for j in range(2*Nphi+2):
			norm=norm+hybridation[i,j]**2
		
	hybridation=hybridation/sqrt(abs(norm))
	
#   DVR transformation
	for i in range(Nx):
		for j in range(2*Nphi+2):
			hybridation[i,j]=hybridation[i,j]/(sqrt(delta_x[i])*sqrt(delta_phi))

#   Checkout the normalization
	norm=0.
	for i in range(Nx):
		for j in range(2*Nphi+2):
			norm=norm+hybridation[i,j]**2*delta_x[i]*delta_phi  

	print('Checkout norm=%g for n=%d'%(norm,n))
	
	THETA, PHI=meshgrid(theta,phi) 
	R=transpose(abs(hybridation))

	return None

	fig = plt.figure(3)
	ax = fig.gca(projection='3d')
	
	X = R * sin(THETA) * cos(PHI)
	Y = R * sin(THETA) * sin(PHI)
	Z = R * cos(THETA)
	surf = ax.plot_surface(X, Y, Z,rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)
	ax.set_xlabel('$x$',size=20)
	ax.set_ylabel('$y$',size=20)
	ax.set_zlabel('$z$',size=20)
	title("$|(\psi_{n}(\\theta,\phi)+\psi'_{n}(\\theta,\phi))/\sqrt{2}| \\rightarrow sp_z $",size=20)
	#ax.set_zlim3d(-1, 1)
	#fig.colorbar(surf)
	plt.show()
	
