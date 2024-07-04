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

from .constants import ci


#*****************************************************************************#        
def Get_k(delta_x,V):
	minloc = argmin(V)
	k = sqrt((V[minloc+1]-2*V[minloc]+V[minloc-1])/delta_x[0]**2)
	print('<br>k=%g in a.u.'%k)

	return k

#****************************************************************************#
def Get_Current(psi,N,mu,delta_x):
	grad_psi = diff(psi,axis=1)/delta_x
	psi_middle = (psi[1:,:]+psi[:-1,:])/2
	J = ic/(2*mu)*(psi*grad_psi.conj()-psi.conj()*grad_psi)
	J[N-1,:] = J[N-2,:]

	return J

def Get_product(Psi,Psi1,x,delta_x):
	if len(Psi.shape)==1:
		Psi = Psi.reshape((1,len(Psi)))
	if len(Psi1.shape)==1:
		Psi1 = Psi1.reshape((1,len(Psi1)))

	return (conj(Psi)*Psi1) @ delta_x

def Get_Observables(Psi,Psi1,x,delta_x,V=None,dV_dx=None,Eshift=0):
	#print('<br>Get_Observables...')
	if len(Psi.shape)==1:
		Psi = Psi.reshape((1,len(Psi)))
	if len(Psi1.shape)==1:
		Psi1 = Psi1.reshape((1,len(Psi1)))

	norm = abs(conj(Psi)*Psi) @ delta_x

	E = Eshift+(conj(Psi)*Psi1) @ delta_x #Psi1 <==> H|Psi>
	E = sign(real(E))*abs(E) # need to be checked for energy

	if dV_dx is None and V is None:
		return norm,E
	else:
		#dV_dx_average	= ((conj(Psi[:,1:-1])*Psi[:,1:-1])*dV_dx[1:-1]) @ MidPointsOperator(delta_x[1:-1])
		if V is not None:
			V_average	= ((conj(Psi)*Psi)*V) @ delta_x
			V_average	= sign(real(V_average))*abs(V_average)+Eshift
		else:
			V_average = None
		if V is not None:
			dV_dx_average	= ((conj(Psi)*Psi)*dV_dx) @ delta_x
			dV_dx_average	= sign(real(dV_dx_average))*abs(dV_dx_average)
		else:
			dV_dx_average = None

		return norm,E, V_average, dV_dx_average 


def Get_Observables2D(psi,psi1,delta_x,delta_y,Eshift=0):
	if len(psi.shape)==1:
		psi = psi.reshape((1,len(psi)))
	if len(psi1.shape)==1:
		psi1 = psi1.reshape((1,len(psi1)))

	dX,dY	= np.meshgrid(delta_x,delta_y, sparse=False, indexing='ij')
	dS		= (dX*dY).flatten()

	norm	= abs( conj(psi)*psi @ dS )

	E		= Eshift + (conj(psi)*psi1) @ dS
	E		= sign(real(E))*abs(E) # need to be checked for energy

	return norm,E

def Get_Observables2Dsingle(psi,psi1,delta_x,delta_y,Eshift=0):
	dx,dy	= delta_x,delta_y

	norm	= abs( dx @ (conj(psi)*psi) @ dy )

	E		= Eshift + dx @ (conj(psi)*psi1) @ dy
	E		= sign(real(E))*abs(E) # need to be checked for energy

	return norm,E


#   Calculation of rotational constants B_v=<v|1/(2m*R^2)|v>
def Get_Rotational_Constant(psi,mu,V,r,delta_r):
	k0 = Get_k(delta_r,V) 	# k0 is the spring force constant in the harmonic approximation

	if len(psi.shape)==1:
		psi = psi.reshape((1,len(psi)))
	pp		= conj(psi)*psi

	# Integrate Trapeze
	r_average	= pp @ (r*delta_r)
	r[r==0]	= 1e-4
	Bv		= pp @ (delta_r/(r**2)/(2*mu))
	Dv		= pp @ (delta_r/(r**6)/(2*mu*k0))

	return Bv,Dv,r_average

#*****************************************************************************#
def Get_Heisenberg_Uncertainty(psi,x,delta_x0,delta_x,kx=None,delta_kx=None,psi_FT=None,Type=0):
	#print('<br>Get_Heisenberg_Uncertainty Type=',Type)
	#print('<br>psi.shape=',psi.shape)
	#print('<br>x.shape=',x.shape)
	#print('<br>delta_x0.shape=',delta_x0.shape)
	#print('<br>delta_x.shape=',delta_x.shape)
	if len(psi.shape)==1:
		psi = psi.reshape((1,len(psi)))
	pp		= conj(psi)*psi

	# Integrate Trapeze
	x_average		= pp @ (x*delta_x)
	x_average		= sign(real(x_average))*abs(x_average)

	# Integrate Trapeze
	xsquare_average	= abs(pp @ (x**2*delta_x))

	# Momentum
	if psi_FT is None:
		grad_psi,laplacien_psi	= Derivatives2(psi, delta_x0, delta_x, Type)
		# Integrate Trapeze
		px_average				= (conj(psi)*grad_psi) @ delta_x
		pxsquare_average		= abs( (conj(psi)*laplacien_psi) @ delta_x )
	else:
		if len(psi_FT.shape)==1:
			psi_FT = psi_FT.reshape((1,len(psi_FT)))
		pp_FT	= conj(psi_FT)*psi_FT

		if isscalar(delta_kx):
			dk		= MidPointsOperator(np.diff(kx))
		else:
			dk		= delta_kx

		# Integrate Trapeze
		px_average		= pp_FT @ (kx*dk)
		# Integrate Trapeze
		pxsquare_average	= abs( pp_FT @ (kx**2*dk) )

	px_average			= sign(real(px_average))*abs(px_average) # -ci*...

	Delta_X		= sqrt(xsquare_average-x_average**2)
	Delta_Px	= sqrt(pxsquare_average-px_average**2)
	Heisenberg	= Delta_X*Delta_Px

	return Heisenberg,x_average,px_average,Delta_X,Delta_Px


def Get_Heisenberg_Uncertainty2D(psi,x,delta_x0,delta_x,y,delta_y0,delta_y,kx=None,delta_kx=None,ky=None,delta_ky=None,psi_FT=None,Type=0):
	#print('<br>Get_Heisenberg_Uncertainty2D Type=',Type)
	#print('<br>psi.shape=',psi.shape)
	#print('<br>x.shape=',x.shape)
	#print('<br>delta_x=',delta_x)
	#print('<br>y.shape=',y.shape)
	#print('<br>delta_y=',delta_y)
	if len(psi.shape)==1:
		psi = psi.reshape((1,len(psi)))
	pp		= conj(psi)*psi

	dX,dY	= np.meshgrid(delta_x,delta_y, sparse=False, indexing='ij')
	dS		= (dX*dY).flatten()
	X,Y		= np.meshgrid(x,y, sparse=False, indexing='ij')
	X,Y		= X.flatten(),Y.flatten()

	x_average		= pp @ (X*dS)
	y_average		= pp @ (Y*dS)
	xsquare_average	= abs( pp @ (X**2*dS) )
	ysquare_average	= abs( pp @ (Y**2*dS) )
	#print('<br>x_average.shape=',x_average.shape)

	# Momentum
	if psi_FT is None:
		psiy = psi.reshape((psi.shape[0],len(x),len(y)))
		psix = np.moveaxis(psiy,[0,1,2],[0,2,1])

		grad_psix,laplacien_psix	= Derivatives2(psix, delta_x0, delta_x, Type) # -laplacien_psix
		grad_psiy,laplacien_psiy	= Derivatives2(psiy, delta_y0, delta_y, Type) # -laplacien_psiy
		px_average				= ((conj(psix)*grad_psix) @ delta_x) @ delta_y
		pxsquare_average		= abs( ((conj(psix)*laplacien_psix) @ delta_x) @ delta_y )
		py_average				= ((conj(psiy)*grad_psiy) @ delta_y) @ delta_x
		pysquare_average		= abs( ((conj(psiy)*laplacien_psiy) @ delta_y) @ delta_x )
	else:
		pp_FT = conjugate(psi_FT)*psi_FT

		if isscalar(delta_kx):
			dkx		= MidPointsOperator(np.diff(kx))
		else:
			dkx		= delta_kx
		if isscalar(delta_ky):
			dky		= MidPointsOperator(np.diff(ky))
		else:
			dky		= delta_ky

		dKX,dKY	= np.meshgrid(dkx,dky, sparse=False, indexing='ij')
		dSk		= (dKX*dKY).flatten()
		KX,KY	= np.meshgrid(kx,ky, sparse=False, indexing='ij')
		KX,KY	= KX.flatten(),KY.flatten()

		px_average		= abs( pp_FT @ (KX*dSk) )
		py_average		= abs( pp_FT @ (KY*dSk) )
		pxsquare_average	= abs( pp_FT @ (KX**2*dSk) )
		pysquare_average	= abs( pp_FT @ (KY**2*dSk) )


	Delta_X			= sqrt(xsquare_average-x_average**2)
	Delta_Y			= sqrt(ysquare_average-y_average**2)
	Delta_Px		= sqrt(pxsquare_average-px_average**2)
	Delta_Py		= sqrt(pysquare_average-py_average**2)
	Heisenberg_X	= Delta_X*Delta_Px
	Heisenberg_Y	= Delta_Y*Delta_Py
	Heisenberg		= Heisenberg_Y+Heisenberg_X

	return Heisenberg,Heisenberg_X,Heisenberg_Y,x_average,px_average,Delta_X,Delta_Px,y_average,py_average,Delta_Y,Delta_Py

def Get_Heisenberg_Uncertainty2Dsingle(psi,x,delta_x0,delta_x,y,delta_y0,delta_y,kx=None,delta_kx=None,ky=None,delta_ky=None,psi_FT=None,Type=0):
	#print('<br>Get_Heisenberg_Uncertainty2Dsingle Type=',Type)

	#print('<br>psi.shape=',psi.shape)
	#print('<br>x.shape=',x.shape)
	#print('<br>delta_x.shape=',delta_x.shape)
	#print('<br>y.shape=',y.shape)
	#print('<br>delta_y.shape=',delta_y.shape)

	pp		= conj(psi)*psi
	dx,dy	= delta_x,delta_y

	x_average		= (x*dx) @ pp @ dy
	y_average		=     dx @ pp @ (y*dy)
	xsquare_average	= abs( (x**2*dx) @ pp @ dy )
	ysquare_average	= abs(        dx @ pp @ (y**2*dy) )

	# Momentum
	if psi_FT is None:
		psiy = psi
		psix = np.moveaxis(psi,[0,1],[1,0])

		grad_psix,laplacien_psix	= Derivatives2(psix, delta_x0, delta_x, Type) # -laplacien_psix
		grad_psiy,laplacien_psiy	= Derivatives2(psiy, delta_y0, delta_y, Type) # -laplacien_psiy
		px_average				= ((conj(psix)*grad_psix) @ delta_x) @ delta_y
		pxsquare_average		= abs( ((conj(psix)*laplacien_psix) @ delta_x) @ delta_y )
		py_average				= ((conj(psiy)*grad_psiy) @ delta_y) @ delta_x
		pysquare_average		= abs( ((conj(psiy)*laplacien_psiy) @ delta_y) @ delta_x )
	else:
		#print('<br>psi_FT.shape=',psi_FT.shape)
		#print('<br>kx.shape=',kx.shape)
		#print('<br>delta_kx.shape=',delta_kx.shape)
		#print('<br>ky.shape=',ky.shape)
		#print('<br>delta_ky.shape=',delta_ky.shape)

		pp_FT = conjugate(psi_FT)*psi_FT
		dkx,dky	= delta_kx,delta_ky

		px_average			= abs( (kx*dkx) @ pp_FT @ dky )
		py_average			= abs(      dkx @ pp_FT @ (ky*dky) )
		pxsquare_average	= abs( (kx**2*dkx) @ pp_FT @ dky )
		pysquare_average	= abs(         dkx @ pp_FT @ (ky**2*dky) )


	Delta_X				= sqrt(xsquare_average-x_average**2)
	Delta_Y				= sqrt(ysquare_average-y_average**2)
	Delta_Px			= sqrt(pxsquare_average-px_average**2)
	Delta_Py			= sqrt(pysquare_average-py_average**2)
	Heisenberg_X		= Delta_X*Delta_Px
	Heisenberg_Y		= Delta_Y*Delta_Py
	Heisenberg			= Heisenberg_Y+Heisenberg_X

	return Heisenberg,Heisenberg_X,Heisenberg_Y,x_average,px_average,Delta_X,Delta_Px,y_average,py_average,Delta_Y,Delta_Py



#*****************************************************************************#
def Derivatives2(f,dx0,dx,Type=0, order=2):
	#print('<br>Derivative',order,' Type=',Type)
	Nshape = f.shape
	N = Nshape[-1]-1
	Naxis = len(Nshape)-1

	#if dx0 is not None: print('<br>Derivative1 dx0.shape=',dx0.shape)
	#if dx is not None: print('<br>Derivative1 dx.shape=',dx.shape)
	#print('<br>Derivative1 f.shape=',f.shape)

	if Type==0:
		#O = np.zeros((append(Nshape[:-1],[[1]])))
		#deltay = np.diff(f,axis=Naxis)
		#dy_dx = deltay/dx0[0]
		#dy_dx_out = np.concatenate([dy_dx,O],axis=Naxis)
		deltay = f.take(np.concatenate((range(1,N+1),[0],)),axis=Naxis) - f
		dy_dx = deltay/dx0
		dy_dx_out = (dy_dx.take(np.concatenate(([N],range(N))),axis=Naxis)+dy_dx)/2
	elif Type==1:
		if Naxis==0:
			O = [[0]]
		else:
			O = np.zeros((append(Nshape[:-1],[[1]])))
		deltay = np.concatenate([f,O],axis=Naxis) - np.concatenate([O,f],axis=Naxis)
		dy_dx = deltay/dx0
		dy_dx_out = midPoints(dy_dx)
	elif Type==2 or Type==3:
		deltay = np.diff(f, axis=Naxis)
		dy_dx = deltay/dx0
		O = np.zeros((append(Nshape[:-1],[[1]])))
		dy_dx_out = np.concatenate([dy_dx,O],axis=Naxis)

	#	print('<br>Derivative1 dx0.shape=',dx0.shape)
	#	print('<br>Derivative1 deltay.shape=',deltay.shape)
	#	print('<br>Derivative1 dy_dx.shape=',dy_dx.shape)
	#	print('<br>Derivative1 dy_dx_out.shape=',dy_dx_out.shape)
	
	if order==1:
		return dy_dx_out

	#print('<br>Derivative2...')
	if Type==0:
		#d2y_dx2_out = np.concatenate([O,np.diff(dy_dx, axis=Naxis),O],axis=Naxis)/dx0[0]
		#d2y_dx2_out = ( dy_dx.take(np.r_[N,0:N],axis=Naxis) - dy_dx)/dx
		d2y_dx2_out = ( dy_dx.take(np.r_[1:N+1,0],axis=Naxis) - dy_dx)/dx
	elif Type==1:
		d2y_dx2_out = np.diff(dy_dx, axis=Naxis)/dx
	elif Type==2 or Type==3:
		d2y_dx2_out = np.concatenate([O,np.diff(dy_dx, axis=Naxis)/dx[1:-1],O],axis=Naxis)

	#	print('<br>Derivative1 d2y_dx2_out.shape=',d2y_dx2_out.shape)

	return dy_dx_out,d2y_dx2_out


def midPoints(x):
	Nshape = x.shape
	N = Nshape[-1]
	Naxis = len(Nshape)-1

	return (x.take(range(1,N),axis=Naxis) + x.take(range(0,N-1),axis=Naxis))/2

def MidPointsOperator(dx):
	return ( np.concatenate([[0],dx]) + np.concatenate([dx,[0]]) )/2
