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

from math import *
from numpy import *
import numpy as np
from Quantum_mechanics import au_to_amu,au_to_fm,au_to_MeV,au_to_sec

#*************************************************************# 
def CAP(x,LCAP,xL,A2):
	CAP = np.zeros(x.shape,float)
	condition = logical_and(x>=(xL-LCAP), x<=xL)
	xbar = (x[condition]-(xL-LCAP))/LCAP
	CAP[condition] = 3/2*A2*xbar**2

	return CAP

#*************************************************************#
def Get_CAP_Magnetude(Ekin_min,Ekin_max,m,LCAP):
	cc		= np.array((2.88534, -0.0628886, -0.0853079, 0.0133969, -0.00078675, 1.65013e-5))
	func_l	= lambda l : cc[5]*l**5+cc[4]*l**4+cc[3]*l**3+cc[2]*l**2+cc[1]*l+cc[0]

	if Ekin_max==None:
		Ekin_average		= Ekin_min
	else:
		Ekin_average		= Ekin_min**0.6242 * Ekin_max**0.3759

	lambda_dB_average	= 2*pi/sqrt(2*m*Ekin_average)

	L_lambda			= np.linspace(0.001,15,100)
#	A2_E				= func_l(L_lambda)
	A2					= func_l(LCAP/lambda_dB_average)*Ekin_average

	return A2,Ekin_average,lambda_dB_average


#*************************************************************#
def Get_v_leackage(r,V,E,v_loc,maxloc):
	print("<br><br>Get_v_leackage:")

	v_leackage = v_loc[real(E[v_loc])>0]
	print("<br>v_leackage=",v_leackage)
	i1 = []
	i2 = []
	r1 = []
	r2 = []
	rw = []
	for ind in v_leackage:
		try:
#			i1_tmp, = np.where(np.logical_and(r<r[maxloc], real(E[ind])>V))[-1]
#			i2_tmp, = np.where(np.logical_and(r>r[maxloc], real(E[ind])>V))[0]
#			i1_tmp = (np.logical_and(r<r[maxloc], real(E[ind]-V)>0).nonzero()[0])[-1]+1
			i1_tmp = (np.logical_and(r<r[maxloc], real(E[ind]-V)>0).nonzero()[0])[-1]
			i2_tmp = (np.logical_and(r>r[maxloc], real(E[ind]-V)>0).nonzero()[0])[0]
			print("<br>approximated position (indices) of the intersection of V with localised leackage (quasi-localized) eigenstate (v=",ind,"): i1=",i1_tmp,", i2=",i2_tmp)

			i1.append(i1_tmp)
			i2.append(i2_tmp)
			r1.append(r[i1_tmp])
			r2.append(r[i2_tmp])
			rw.append((r[(real(E[ind]-V)>0)])[0])

		except:
			''' not a localised state '''
			print("<br>not a localised state")

	return rw,r1,r2,i2,i1,v_leackage
