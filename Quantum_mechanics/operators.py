#******************************************************************************#
#
#   Schrodinger_1D is the main program for solving schrodinger equation 
#          using Discret variable representation (DVR) 
#                and Fourier Grid Hmiltonian (FGH) methods 
#
#   Discussion:
#    Prticle submitted to differnet potential square-well, barrier  
#    harmonic, H2 molecule and so on..
#
#
#
#   Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#   Modified:
#
#    January 17th 2017
#
#   Author:
#
#    MEHDI AYOUZ LGPM CENTRALESUPELEC
#    mehdi.ayouz@centralesupelec.fr
#    Bat DUMAS C421. 0141131603
#
#   Optimized and improved for package usage by Alexander V. KOROVIN
#    a.v.korovin73@gmail.com
#
#   input:
#   N, x0, xL, mu, V0
#  
#   output : 
#   E[0:N-1] (eigen energies)
#   psi[0:N-1,0:N-1] (normalized eigen fonctions along x[0:N-1] as row 
#                      and for each eigen value as columne    
#******************************************************************************#
from __future__ import division
import numpy as np
#from numpy import linalg as LA
from pylab import *
from scipy.linalg import toeplitz

from .constants import ci,au_to_fm,au_to_MeV,au_to_amu,au_to_sec
from .CAP import CAP,Get_CAP_Magnetude


def Get_Potential_Operator(x,InputData,Pot_Type=0):
	print('<br>V.....')
	k = 1e5
	if Pot_Type==0: # Infinite
		V0		= InputData['V0']		if 'V0'		in InputData	else 1e3
		V = np.zeros(x.shape,float)
		V[[0,-1]] = V0

	elif Pot_Type==1: # Rectangular
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		x0		= InputData['x0']		if 'x0'		in InputData	else -1
		xL		= InputData['xL']		if 'xL'		in InputData	else 1
		V = V0*(tanh(k*(x-xL))-tanh(k*(x-x0)))/2
		V[[0,-1]] = k

	elif Pot_Type==2 or Pot_Type==3: # N rectangular wells (Double wells, Double steps well)
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		x0		= InputData['x0']		if 'x0'		in InputData	else -1
		xL		= InputData['xL']		if 'xL'		in InputData	else 1
		V = np.zeros(x.shape,float)
		for n,v0 in enumerate(V0) :
			V += Get_Potential_Operator(x,{'V0':V0[n],'x0':x0[n],'xL':xL[n]},Pot_Type=1)

	elif Pot_Type==5 or Pot_Type==24: #harmonic oscillator potential
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		xc		= InputData['xc']		if 'xc'		in InputData	else 0
		print("<br>V0=",V0)
		# V0 = 0.5*(omega)**2
		# V0 = 0.5*(omega)**2*mu
		V = V0*(x-xc)**2

#		if Pot_Type==24: #harmonic oscillator potential (barrier)
#			V = -V0*(x-xc)**2

	elif Pot_Type==6: # Delta potentiel
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		xc		= InputData['xc']		if 'xc'		in InputData	else 0
		# xc as center of well
		sigma	= InputData['sigma']	if 'sigma'	in InputData	else 1
		V = V0/(sigma*sqrt(pi))*exp(-(x-xc)**2/sigma**2)#-ci*CAP(x[i])
		V[[0,-1]] = k

	elif Pot_Type==7: # N delta potentiel (Double Delta potentiel)
		V0		= InputData['V0']		if 'V0'	in InputData	else 1
		xc		= InputData['xc']		if 'xc'	in InputData	else 0
		# xc as array of centres of wells
		sigma	= InputData['sigma']	if 'sigma'	in InputData	else 1
		V = np.zeros(x.shape,float)
		for n,xc_n in enumerate(xc) :
			V += Get_Potential_Operator(x,{'V0':V0,'xc':xc_n,'sigma':sigma},Pot_Type=6)

#	elif Pot_Type==8: # H2

#	elif Pot_Type==9: # H2+

	elif Pot_Type==11: # Hydrogenoid Atoms
		Z		= InputData['Z']		if 'Z'		in InputData	else 1
		V = -Z/x

	elif Pot_Type==12: # alpha decay
		V0		= InputData['V0']		if 'V0'		in InputData	else -115
		V1		= InputData['V1']		if 'V1'		in InputData	else 500
		Z		= InputData['Z']		if 'Z'		in InputData	else 84
		delta	= InputData['delta']	if 'delta'	in InputData	else 1
		sigma	= InputData['sigma']	if 'sigma'	in InputData	else 0.8
		LCAP	= InputData['LCAP']		if 'LCAP'	in InputData	else 30
		xL		= InputData['xL']		if 'xL'		in InputData	else 200
		A		= InputData['A']		if 'A'		in InputData	else 212
		A2		= InputData['A2']		if 'A2'		in InputData	else 1
		R		= InputData['R']		if 'R'		in InputData	else 1.07*(4**(1/3)+(InputData.A-4)**(1/3)) # Radius of alpha particle and daughter nucleus in fm

		V_WS    = np.zeros(x.shape,float)
		V_SR    = np.zeros(x.shape,float)
		V_LR    = np.zeros(x.shape,float)
		V_Model = np.zeros(x.shape,float)

		Vm=-66       #in MeV

		r_fm = x*au_to_fm

		condition = x<=R
		V_SR[condition]    = 2*(Z-2)*(3*R**2-x[condition]**2)/(2*R**3)
		V_Model[condition] = Vm/au_to_MeV

		condition = np.invert(condition)
		V_Model[condition] = 2*(Z-2)/x[condition]
		V_LR[condition]    = 2*(Z-2)/x[condition]

		V_WS   = V0/(1+exp((x-R)/sigma))
		V_Wall = V1*exp(-r_fm/delta)
		V      = V_Wall+V_WS+V_SR+V_LR-ci*CAP(x,LCAP,xL,A2)

		minval = min(real(V))
		minloc = nonzero(real(V)==minval)[0].reshape(())
		rmin   = x[minloc]
		print("<br>minval=",minval)
		print("<br>minloc=",minloc)
		print("<br>rmin=",rmin)

		maxval = max(real(V[minloc:]))
		maxloc = nonzero(real(V)==maxval)[0].reshape(())
		rmax   = x[maxloc]
		print("<br>maxloc=",maxloc)
		print("<br>rmax=",rmax)

		return V,V_Wall,V_WS,V_LR,V_SR,minloc,maxloc,minval,maxval,rmax,rmin,V_Model

	elif(Pot_Type==13): # Morse potential for simulating diatomics
		xc		= InputData['xc']		if 'xc'		in InputData	else 0
		De		= InputData['De']		if 'De'		in InputData	else 10
		alpha	= InputData['alpha']	if 'alpha'	in InputData	else 0.5
		V = De*(1-exp(-alpha*(x-xc)))**2

	elif(Pot_Type==14): # Ammonia molecule
		xc		= InputData['xc']		if 'xc'		in InputData	else 0
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		De		= InputData['De']		if 'De'		in InputData	else 0.04730#0.04764
		c		= InputData['c']		if 'c'		in InputData	else 0.05684
		alpha	= InputData['alpha']	if 'alpha'	in InputData	else 1.3696
		# xc?
		V = V0*(x-xc)**2+c*exp(-alpha*(x-xc)**2)-De

	elif(Pot_Type==20):
		V = np.zeros(x.shape,float)

	elif(Pot_Type==21): #barrier potential
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		xc		= InputData['xc']		if 'xc'		in InputData	else 0
		dx		= InputData['dx']		if 'dx'		in InputData	else 1e-2
		V = V0*tanh((x-xc)/dx)/2+V0/2

	elif(Pot_Type==22): #rectangular barrier potential
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		xc		= InputData['xc']		if 'xc'		in InputData	else 0
		dx		= InputData['dx']		if 'dx'		in InputData	else 1e-2
		d		= InputData['d']		if 'd'		in InputData	else 1
		V = V0*(tanh((x-(xc-d/2))/dx)-tanh((x-(xc+d/2))/dx))/2

	elif(Pot_Type==23): #delta-like potential
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		xc		= InputData['xc']		if 'xc'		in InputData	else 0
		sigma	= InputData['sigma']	if 'sigma'	in InputData	else 1
		V = V0/(sigma*sqrt(pi))*exp(-(x-xc)**2/sigma**2)

	else:
		print("Uknown Pot_Type type (use Pot_Type=0 for .. and Pot_Type=1 for..), Pot_Type=",Pot_Type)

	return V




def Get_Potential_Operator2D(x,y,InputData,Pot_Type=5):

	if Pot_Type==5 or Pot_Type==24: #harmonic oscillator potential
		V0x		= InputData['V0x']		if 'V0x'	in InputData	else 1
		V0y		= InputData['V0y']		if 'V0y'	in InputData	else 1
		xc		= InputData['xc']		if 'xc'		in InputData	else 0
		yc		= InputData['yc']		if 'yc'		in InputData	else 0
		print("<br>V0x=",V0x)
		print("<br>V0y=",V0x)
		#print('<br>xc',xc)
		#print('<br>yc',yc)
		# V0 = 0.5*(omega)**2*mu

		#V = np.zeros((len(x),len(y)),float)
		#for i in range(len(x)):
			#for j in range(len(y)):
				#V[i,j] = V0x*(x[i]-xc)**2+V0y*(y[j]-yc)**2

		VX,VY = np.meshgrid(V0x*(x-xc)**2,V0y*(y-yc)**2, sparse=False, indexing='ij')
		V = VX+VY

	elif(Pot_Type==20):
		#V0		= InputData['V0']		if 'V0'	in InputData	else 1
		Nx = len(x)
		Ny = len(y)
		#V = V0*np.ones((Nx,Ny),float)
		V = np.zeros((Nx,Ny),float)
	elif(Pot_Type==31):
		V0		= InputData['V0']		if 'V0'	in InputData	else 100
		l		= InputData['l']		if 'l'	in InputData	else 5
		ls		= InputData['ls']		if 'ls'	in InputData	else 10
		xc		= InputData['xc']		if 'xc'	in InputData	else 0
		yc		= InputData['yc']		if 'yc'	in InputData	else 0
		Nx = len(x)
		Ny = len(y)
		print('<br>l=',l)
		print('<br>xc=',xc)
		print('<br>ls=',ls)
		print('<br>yc=',yc)
		V = np.zeros((Nx,Ny),float)
		X,Y = np.meshgrid(x-xc,y-yc, sparse=False, indexing='ij')
		condition = np.logical_and(abs(X)<l/2, abs(Y)>ls/2)
		#condition = np.logical_and(abs(X)<xc+l/2, abs(Y)>yc+ls/2)
		V[condition] = V0*exp(-abs(X[condition]))
	elif(Pot_Type==32):
		V0		= InputData['V0']		if 'V0'	in InputData	else 100
		l		= InputData['l']		if 'l'	in InputData	else 8
		ls		= InputData['ls']		if 'ls'	in InputData	else 4
		ly		= InputData['ly']		if 'ly'	in InputData	else 4
		xc		= InputData['xc']		if 'xc'	in InputData	else 0
		yc		= InputData['yc']		if 'yc'	in InputData	else 0
		Nx = len(x)
		Ny = len(y)
		V = np.zeros((Nx,Ny),float)
		X,Y = np.meshgrid(x-xc,y-yc, sparse=False, indexing='ij')
		condition = np.logical_or( np.logical_and(abs(X)<l/2, abs(Y)>ly/2),abs(Y)>ly/2+ls )
		V[condition] = V0*exp(-abs(X[condition]))
	elif(Pot_Type==33):
		V0		= InputData['V0']		if 'V0'	in InputData	else 100
		l		= InputData['l']		if 'l'	in InputData	else 8
		ls		= InputData['ls']		if 'ls'	in InputData	else 4
		ly		= InputData['ly']		if 'ly'	in InputData	else 4
		xc		= InputData['xc']		if 'xc'	in InputData	else 0
		yc		= InputData['yc']		if 'yc'	in InputData	else 0
		Nx = len(x)
		Ny = len(y)
		V = np.zeros((Nx,Ny),float)
		X,Y = np.meshgrid(x-xc,y-yc, sparse=False, indexing='ij')
		condition = np.logical_or( np.logical_and(abs(X)<l/2, abs(Y)>ls/2),abs(Y)>ls/2+ly )
		V[condition] = V0*exp(-abs(X[condition]))

	else:
		print("Uknown type Pot_Type=",Pot_Type)

	return V





def Get_E_min_max(V,delta_x,mu):
	pmax = pi/delta_x.min()
	Emax = pmax**2/(2.*mu)+V.max()
	Emin = V.min()
	Eshift = (Emax+Emin)/2

	return Emax,Emin,pmax,Eshift

def Get_E_min_max2D(V,delta_x,delta_y,mu):
	pxmax = pi/delta_x.min()
	pymax = pi/delta_y.min()
	Emax = (pxmax**2+pymax**2)/(2.*mu)+V.max()
	Emin = V.min()
	Eshift = (Emax+Emin)/2

	return Emax,Emin,pxmax,pymax,Eshift



def Get_Potential_Operator_orbital(r,l,mu):
	print("<br>1 mu=",mu)
	print("<br>1 l=",l)
	r[r==0] = 1e-4
	return l*(l+1)/(2*mu*r**2)
def Get_Potential_Operator_spinorbite(r,s,l,J,me,e):
	c = 137
#	me = 1
#	e = 1
	return e**2*(J*(J+1)-l*(l+1)-s*(s+1))/(2*me**2*c**2*r**3)


def Get_Potential_Operator_from_file(fname):
	print('fname=',fname)
	data= open(fname, "r")
	lines=data.readlines()
	N = len(lines)

	x = np.zeros(N,float)
	V = np.zeros(N,float)
	for i,line in enumerate(lines):
		p = line.split()
		x[i] = float(p[0])
		V[i] = float(p[1])
	data.close()

	return V,x,x.min(),x.max(), np.diff(x)

#*****************************************************************************#        
def Get_Kinetic_Operator(x,mu,LBOXx,Type):
	print('<br>LBox=',LBOXx)
	print('<br>x.min()=',x.min())
	print('<br>x.max()=',x.max())
	#LBOXx= 0.00377945037811637

	print('<br>Type=',Type)

	N = len(x)
	if(Type==0):
		coef= (pi/LBOXx)**2/mu
		Di	= np.arange(1,N)
		if(N%2==0):
			T_diag		= (N*N+2)/6.
			T_nondiag	= (-1.)**Di/(sin(Di*pi/N))**2
			#print('<br>even T=',coef*hstack( (T_diag, T_nondiag) ))
		else:
			T_diag		= (N*N-1)/6.
			T_nondiag	= (-1.)**Di*cos(Di*pi/N)/(sin(Di*pi/N))**2
		T = toeplitz( coef*hstack( (T_diag, T_nondiag) ) )
	elif(Type==1):
		# sin basis set
		coef	= (pi/LBOXx)**2/(2*mu)/2
		T = np.zeros((N,N),float)
		I1, I2 = meshgrid(np.arange(N),np.arange(N))
		cind = np.logical_not(I1==I2)
		DI = I1[cind]-I2[cind]
		SI = I1[cind]+I2[cind]
		T[cind] = coef*(-1.)**DI*(1./sin(pi*DI/(2*(N+1)))**2 - 1./sin(pi*(SI+2)/(2*(N+1)))**2)
		cind = (I1==I2)
		T[cind] = coef*((2*(N+1)**2+1)/3. - 1/sin(pi*(I1[cind]+1)/(N+1))**2)
	elif(Type==2):
		i = (np.arange(N)).reshape(len(x),1)
		xx = x.reshape(len(x),1)
		np.seterr(divide='ignore') # it will be correct Inf in diagonal elements
		T = (-1.)**(i.T-i)*(xx.T+xx)/sqrt(xx.T*xx)/(xx.T-xx)**2/(2*mu)
		np.seterr(divide='warn')
		np.fill_diagonal(T, 1./(12*x**2)*(4+(4*N+2)*x-x**2)/(2*mu) ) #-2*(N-1)/3-1/2+x[i]**2/3

	elif(Type==3):
		i = (np.arange(N)).reshape(len(x),1)
		xx = x.reshape(len(x),1)
		np.seterr(divide='ignore') # it will be correct Inf in diagonal elements
		T = (-1.)**(i.T-i)*(2./(xx.T-xx)**2-1./2)/(2*mu) # 1/2-2/(x[i]-x[ip])**2
		np.seterr(divide='warn')
		np.fill_diagonal(T, (4*N-1-2*x**2)/6./(2*mu) ) #-2*(N-1)/3-1/2+x[i]**2/3

	return T

def Get_Kinetic_Operator2D(x,y,mu,LBOXx,LBOXy,Type=0):
	return Get_Kinetic_Operator(x,mu,LBOXx,Type), Get_Kinetic_Operator(y,mu,LBOXy,Type)

#*****************************************************************************#
def Get_Hamiltonian_Operator(T,V):
	return T+diag(V)

def Get_Hamiltonian_Operator2D(Nx,Ny,Tx,Ty,V):
#	return   (np.einsum('ij,kl->ljki', Tx, np.eye(Ny,Ny))+\
#			np.einsum('ij,kl->ikjl', Ty, np.eye(Nx,Nx))).reshape((Nx*Ny,Nx*Ny))+\
#			np.diag((V.T).flatten())
	return   (np.einsum('ij,kl->ikjl', Tx, np.eye(Ny,Ny))+\
			np.einsum('ij,kl->ljki', Ty, np.eye(Nx,Nx))).reshape((Nx*Ny,Nx*Ny))+\
			np.diag(V.flatten()) # H[xy,xy]


#*************************************************************#
def Get_Ordered_Eigen_States(P,E):
	ind = np.argsort(real(E))

	return E[ind], P[:,ind]

