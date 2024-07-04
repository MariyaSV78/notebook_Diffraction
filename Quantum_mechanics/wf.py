#******************************************************************************#
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
#******************************************************************************#
from __future__ import division
import numpy as np
from pylab import *

from .constants import ci
from .observables import MidPointsOperator

from .common import func_timetest

#*****************************************************************************#    
def Get_Wave_Functions(psi,delta_x):
	psiModulus = conj(psi)*psi
	Norme = np.sum(Repartition,axis=0)
	Repartition = np.cumsum(psiModulus,axis=0)*delta_x

	return Repartition,Norme,psiModulus

def Get_Wave_Functions_Localization(E,psi,x,delta_x,N,Vmax,xmax):
	condition_x = x<xmax
	dx = delta_x[condition_x]
	Ratio = abs( (psi[:,condition_x]*psi[:,condition_x])@dx )*100
	#Ratio = abs( (conjugate(psi[:,condition_x])*psi[:,condition_x])@dx )*100

	ind, = np.where(np.logical_and(Ratio>90,real(E)<real(Vmax)))
	v_times_v  = abs( (psi[ind,:]*psi[ind,:])@  delta_x )
	v_times_vp = abs( (psi[ind,:]*psi[ind+1,:])@delta_x )
	#v_times_v  = abs( (conjugate(psi[ind,:])*psi[ind,:])@  delta_x )
	#v_times_vp = abs( (conjugate(psi[ind,:])*psi[ind+1,:])@delta_x )
	v = np.arange(N)[ind]
	#print('<br><br> v localized \t <psi_v|psi_v> \t <psi_v|psi_v+1>')
	print('<br><br> v_localized \t psi_v|psi_v \t psi_v|psi_v+1')
	for n in range(len(v)):
		print('<br> ',v[n],'\t',v_times_v[n],'\t',v_times_vp[n])

	return Ratio,v

def Get_Wave_Functions_Normalization(P,delta_x): # Pv,x
	if len(P.shape)==1:
		P = P.reshape((1,len(P))) 
	norm	= (conj(P)*P) @ delta_x
	max_val	= P[(np.arange(P.shape[0]),np.argmax(abs(P),axis=1))];
	#       Setting bound wavefunctions to be real and norm
	return np.einsum( 'vx,v->vx',P,abs(max_val)/max_val/sqrt(norm) )

def Get_Wave_Functions_Normalization2D(P,delta_x,delta_y):
	if len(P.shape)==1:
		P = P.reshape((1,len(P)))

	#dX,dY = np.meshgrid(np.hstack((delta_x,delta_x[-1])),np.hstack((delta_y,delta_y[-1])), sparse=False, indexing='ij')
	dX,dY = np.meshgrid(delta_x,delta_y, sparse=False, indexing='ij')

	norm	= (conj(P)*P) @ (dX*dY).flatten()
	max_val	= P[(np.arange(P.shape[0]),np.argmax(abs(P),axis=1))];

	return np.einsum( 'vx,v->vx',P,abs(max_val)/max_val/sqrt(norm) )
	#return ( (P.T)*abs(max_val)/max_val/sqrt(norm) ).T

#*************************************************************#
#   Fourrier transfrom of psi(x,t) (== \bar{\psi}=int exp(-ikr)psi(r)dr/sqrt(2pi))
def Get_Fourrier_Transform_WF(psi,x,delta_x,kx):
	X,KX = np.meshgrid(x,kx, sparse=False, indexing='ij')

	if len(psi.shape)==1:
		psi = psi.reshape((1,len(psi)))
		psi_FT = ( (psi*delta_x) @ exp(-ci*KX*X) )[0]/sqrt(2*pi) # Integrate Trapeze
	else:
		psi_FT = ( (psi*delta_x) @ exp(-ci*KX*X) )/sqrt(2*pi) # Integrate Trapeze

	return psi_FT

def Get_Fourrier_Transform_WF2D(psi,x,delta_x,y,delta_y,kx,ky):
	KX,X = np.meshgrid(kx,x, sparse=False, indexing='ij')
	KY,Y = np.meshgrid(ky,y, sparse=False, indexing='ij')

	Nx = len(x)
	Ny = len(y)
	Nkx = len(kx)
	Nky = len(ky)
	Neig = psi.shape[0]
	psi_FT = np.einsum('nxy,kx,ly->nkl', psi.reshape((Neig,Nx,Ny)), exp(-ci*(KX*X))*delta_x, exp(-ci*(KY*Y))*delta_y ).reshape((Neig,Nkx*Nky))/(2*pi)

	return psi_FT

# for single eigen state ( psi(x,y) )
def Get_Fourrier_Transform_WF2Dsingle(psi,x,delta_x,y,delta_y,kx,ky):
	KX,X = np.meshgrid(kx,x, sparse=False, indexing='ij')
	KY,Y = np.meshgrid(ky,y, sparse=False, indexing='ij')

	psi_FT = ( (exp(-ci*(KX*X)) * delta_x) @ psi @  (exp(-ci*(KY*Y))*delta_y).T )/(2*pi) # Integrate Trapeze

	return psi_FT

