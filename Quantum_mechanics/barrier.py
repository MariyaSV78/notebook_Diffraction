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
#from numpy import linalg as LA
from pylab import *
from scipy.interpolate import interp1d

from .constants import au_to_fm,au_to_MeV,au_to_sec
from .observables import MidPointsOperator

#*************************************************************#
def Get_Classical_Model(E,mu,v_leackage,psi,rw,r1,i1,i2):
	tauF	= np.zeros(len(v_leackage))
	T		= np.zeros(len(v_leackage))
	for ind,v in enumerate(v_leackage):
		print("<br> Get_Classical_Model: for ind=%i,rw=%i,r1=%i,i1=%i,i2=%i"%(ind,rw[ind],r1[ind],i1[ind],i2[ind]))
		vF			= sqrt(2*abs(E[v])/mu)*au_to_fm*1e-15/au_to_sec
		tauF[ind]	= 2*(r1[ind]-rw[ind])*au_to_fm*1e-15/vF
		nuF			= 1./tauF[ind]
		print("<br>velocity v=%g m/s"%vF)
		print("<br>tauF=%g s nuF=%g 1/s"%(tauF[ind],nuF))
		
		Ain  = max(abs(psi[v,0:i1[ind]]))
		Aout = max(abs(psi[v,i2[ind]:-1]))
		T[ind] = (Aout / Ain)**2;     # probability of alpha particle escaping 
		Gamma = T[ind]*nuF;            # decay constant
		print("<br>Ain=%g Aout=%g T=%g"%(Ain,Aout,T[ind]))
		print('<br>Gamma=%g  1/s'%Gamma)
		print('<br>halftime=%g  s'%(1./Gamma))
		print('<br>ln(2)/Gamma=%g  s'%(log(2)/Gamma))

	return tauF,T

#*************************************************************#
def Get_WKB(E,r,r1,r2,mu,V,delta_r,tauF,v_leackage):
	phase	= np.zeros(len(v_leackage))
	for ind,v in enumerate(v_leackage):
		print("<br> for ind=",ind)
		cind = logical_and(r>r1[ind], r<r2[ind])
		#cind = logical_and(r>r1[ind], r<r2[ind], real(V-E[v])>0)
	#	phase = np.sum(sqrt(2*mu*(real(V[cind]-E[v])))*delta_r[cind])
		phase[ind] =  sqrt(2*mu*(real(V[cind]-E[v]))) @ delta_r[cind]
		#phase[ind] =  sqrt(2*mu*(real(V[cind]-E[v]))) @ MidPointsOperator(np.diff(r[cind]))

		print('<br>phase=%g, <br>T_WK=%g '%(phase[ind],exp(-2*phase[ind])))
		print('<br>Gamma_WKB=%g  1/s'%(exp(-2*phase[ind])/tauF[ind]))
		print('<br>halftime=%g  s'%(tauF[ind]/exp(-2*phase[ind])))
		print('<br>ln(2)/Gamma=%g  s'%(log(2)/(exp(-2*phase[ind])/tauF[ind]))) 

	return phase

#*************************************************************#
def Get_Barrier_Transimission(E,V,mu,n,r,r1,r2,i1,i2,v_leackage,psi,tauF,V_Model,maxloc):
	print("<br>maxloc=",maxloc)
	print("<br>v_leackage=",v_leackage)
	print("<br>i1=",i1)
	print("<br>i2=",i2)

	Total_Transmission	= np.zeros(len(v_leackage))
	Transmission		= np.zeros((len(v_leackage),n))
	ri		 			= np.zeros((len(v_leackage),n+1),float)
	Vi		 			= np.zeros((len(v_leackage),n),float)
	Vi_plot 			= np.zeros((len(v_leackage),n+1,6),float)
	ri_plot 			= np.zeros((len(v_leackage),n+1,6),float)
	for ind,v in enumerate(v_leackage):
		print("<br> for ind=",ind)
		In = int((i2[ind]-i1[ind])/n)
		Delta=r[In]-r[0]

		ri[ind,:]  = np.linspace(r1[ind],r2[ind], num=n+1, endpoint=True)
		print("<br> for r1[ind]=",r1[ind]*au_to_fm)
		print("<br> for r2[ind]=",r2[ind]*au_to_fm)
		print("<br> for ri[ind,:]=",ri[ind,:]*au_to_fm)
		fVi = interp1d(r,V_Model, kind='cubic')
		Vi[ind,:]  = fVi( (ri[ind,0:-1]+ri[ind,1:])/2 )

	#	psi_plot=np.zeros(5000,complex)
	#	psi_plot=interp1d(r, psi[:,v])

		print("<br>Delta=%g fm"%(Delta*au_to_fm)) 

		Transmission[ind,:] = exp(-2*sqrt(2*mu*(Vi[ind,:]-real(E[v])))*Delta)
		print('<br>  Barrier       Height       Transmission')
		for i in range(n):
			Vi_plot[ind,i,:] = [0,0,Vi[ind,i],Vi[ind,i],0,0]
			ri_plot[ind,i,:] = [r.min(),ri[ind,i],ri[ind,i],ri[ind,i+1],ri[ind,i+1],r.max()]
			print('<br>    %d      %g         %g  '%(i,Vi[ind,i]*au_to_MeV,Transmission[ind,i]))
		Total_Transmission[ind] = np.prod(Transmission[ind,:])
		print('<br>Total Transmission for %d segments %g is '%(n,Total_Transmission[ind]))

		print('<br>half-time=%g'%(tauF[ind]/Total_Transmission[ind]))


	return Transmission,Total_Transmission,ri,Vi,ri_plot,Vi_plot

