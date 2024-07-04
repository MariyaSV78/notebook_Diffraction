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
from math import *
from pylab import *
import numpy as np

from .constants import ci
from .observables import MidPointsOperator

#************** Transmission coefficients ************************************#
def Get_Coefficients(N,Psi,A,x,xs,delta_x,k0):
	condition = (x<=xs)
	B = np.sum(Psi[condition]*exp(ci*k0*x[condition])*MidPointsOperator(np.diff(x[condition])))

	return abs(conjugate(B)*B)/abs(conjugate(A)*A)

def Get_Analytic_Coefficients(mu,V0,Ene,d,Pot_Type):
	tmp_Ranalytic = None

	if(Pot_Type==20):
		tmp_Ranalytic=0

	if(Pot_Type==21):
		if(Ene<V0):
			tmp_Ranalytic=1
		elif(Ene>V0): 
			tmp_Ranalytic=((1-sqrt(1-V0/Ene))/(1+sqrt(1-V0/Ene)))**2

	elif(Pot_Type==22):
		if(Ene<V0):
			c=4*Ene*(V0-Ene)
			tmp_Tanalytic=c/(c+V0**2*sinh(sqrt(2*mu*(V0-Ene))*d)**2)
		elif(Ene>V0):
			c=4*Ene*(Ene-V0)
			tmp_Tanalytic=c/(c+V0**2*sin(sqrt(2*mu*(Ene-V0))*d)**2)
		tmp_Ranalytic=1-tmp_Tanalytic

	elif(Pot_Type==23):
		tmp_Tanalytic=1./(1.+mu*V0**2/(2*Ene))

	return tmp_Ranalytic
