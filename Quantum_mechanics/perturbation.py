from __future__ import division
from math import *
import numpy as np
from pylab import *
from scipy import special

#Perturbation_Type
# 0 - no Perturbation
# 1 - elecric field <-- Stark effect
# 2 - delta function
# 3 - magnetic field <-- Zeeman effect
# 4 - Analytical expression
def Get_Perturbation(InputData,x,y=None):
	if not hasattr(InputData, 'Perturbation_Type'):
		InputData.Perturbation_Type = 0
	#	print("<br>Perturbation_Type=",InputData.Perturbation_Type)

	if(InputData.Perturbation_Type==1 or InputData.Perturbation_Type==5 or InputData.Perturbation_Type==6): # elecric field <-- Stark effect
		if not hasattr(InputData, 'perturb_q'):
			InputData.perturb_q			= 1
			print("<br>Perturbation parameter perturb_q is not defined use default perturb_q=1")
		#print("<br>perturb_q=",InputData.perturb_q)
		if not hasattr(InputData, 'perturb_F'):
			InputData.perturb_F			= 1
			print("<br>Perturbation parameter perturb_F is not defined use default perturb_F=1")
		#print("<br>perturb_F=",InputData.perturb_F)

		if(InputData.Perturbation_Type==5):
			if not hasattr(InputData, 'omegap'):
				InputData.omegap			= 1
				print("<br>Perturbation parameter omegap is not defined use default omegap=1")
			print("<br>omegap=",InputData.omegap)
			InputData.Nomegap			= 1
			InputData.Omega_p = InputData.omegap
		elif(InputData.Perturbation_Type==6):
			if not hasattr(InputData, 'Nomegap'):
				InputData.Nomegap			= 1
				print("<br>Perturbation parameter Nomegap is not defined use default Nomegap=1")
			#print("<br>Nomegap=",InputData.Nomegap)
			if not hasattr(InputData, 'omegap_min'):
				InputData.omegap_min			= 1
				print("<br>Perturbation parameter omegap_min is not defined use default omegap_min=1")
			#print("<br>omegap_min=",InputData.omegap_min)
			if not hasattr(InputData, 'omegap_max'):
				InputData.omegap_max			= 1
				print("<br>Perturbation parameter omegap_max is not defined use default omegap_max=1")
			#print("<br>omegap_max=",InputData.omegap_max)
			InputData.Omega_p = linspace(InputData.omegap_min,InputData.omegap_max,InputData.Nomegap)

		if(InputData.Dimensionality==1):
			W = -InputData.perturb_q*InputData.perturb_F*(x-InputData.xc)
		elif(InputData.Dimensionality==2):
			X,Y=np.meshgrid(x,y, sparse=False, indexing='ij')
			W = -InputData.perturb_q*InputData.perturb_F*(X-InputData.xc)
		if(InputData.Perturbation_Type==5 or InputData.Perturbation_Type==6):
			if not hasattr(InputData, 'wt'):
				InputData.wt		= 0
				print("<br>Perturbation parameter wt is not defined use default wt=0")
			#print("<br>w*t=",InputData.wt)
			W *= sin(InputData.wt)
	elif(InputData.Perturbation_Type==2):	# delta function
		if not hasattr(InputData, 'perturb_epsilon'):
			InputData.perturb_epsilon			= 1
			print("<br>Perturbation parameter perturb_epsilon is not defined use default perturb_epsilon=0.01")
		print("<br>perturb_epsilon=",InputData.perturb_epsilon)
		if not hasattr(InputData, 'perturb_eta'):
			InputData.perturb_eta			= 1
			print("<br>Perturbation parameter perturb_eta is not defined use default perturb_eta=1")
		print("<br>perturb_eta=",InputData.perturb_eta)
		if not hasattr(InputData, 'perturb_sigmax'):
			InputData.perturb_sigmax			= 1
			print("<br>Perturbation parameter perturb_sigmax is not defined use default perturb_sigmax=1")
		print("<br>perturb_sigmax=",InputData.perturb_sigmax)
		if not hasattr(InputData, 'perturb_xc'):
			InputData.perturb_xc			= 0
			print("<br>Perturbation parameter perturb_xc is not defined use default perturb_xc=0")
		print("<br>perturb_xc=",InputData.perturb_xc)

		Lx = x.max()-x.min()

		if(InputData.Dimensionality==1):
			W = InputData.perturb_epsilon*InputData.perturb_eta*(Lx/(2*InputData.perturb_sigmax*sqrt(pi)))*exp(-((x-InputData.perturb_xc)/InputData.perturb_sigmax)**2)
		elif(InputData.Dimensionality==2):
			if not hasattr(InputData, 'perturb_sigmay'):
				InputData.perturb_sigmay			= 1
				print("<br>Perturbation parameter perturb_sigmay is not defined use default perturb_sigmay=1")
			print("<br>perturb_sigmay=",InputData.perturb_sigmay)
			if not hasattr(InputData, 'perturb_yc'):
				InputData.perturb_yc			= 0
				print("<br>Perturbation parameter perturb_yc is not defined use default perturb_yc=0")
			print("<br>perturb_yc=",InputData.perturb_yc)
			X,Y=np.meshgrid(x,y, sparse=False, indexing='ij')
			Ly = y.max()-y.min()
	
			W = InputData.perturb_epsilon*InputData.perturb_eta*(Lx/(2*InputData.perturb_sigmax*sqrt(pi)))*exp(-((X-InputData.perturb_xc)/InputData.perturb_sigmax)**2)*(Ly/(2*InputData.perturb_sigmay*sqrt(pi)))*exp(-((Y-InputData.perturb_yc)/InputData.perturb_sigmay)**2)
	
	elif(InputData.Perturbation_Type==3): # magnetic field <-- Zeeman effect
		if not hasattr(InputData, 'perturb_B'):
			InputData.perturb_B			= 1
			print("<br>Perturbation parameter perturb_B is not defined use default perturb_B=1")
		print("<br>perturb_B=",InputData.perturb_B)
		if not hasattr(InputData, 'perturb_mz'):
			InputData.perturb_mz			= 1
			print("<br>Perturbation parameter perturb_mz is not defined use default perturb_mz=1")
		print("<br>perturb_mz=",InputData.perturb_mz)
		me=1
		mu_B=1/(2*me)
		
		W = -InputData.perturb_mz*InputData.perturb_B*mu_B*np.ones(InputData.Nx,float)

	elif(InputData.Perturbation_Type==4): # Analytical expression
		print("<br>Analytical perturbation")

		if not hasattr(InputData, 'strAnalytPerturbationProfile'):
			InputData.strAnalytPerturbationProfile			= '0.1*x'
		print("<br>strAnalytPerturbationProfile=",InputData.strAnalytPerturbationProfile)

		if(InputData.Dimensionality==2):
			x,y=np.meshgrid(x,y, sparse=False, indexing='ij')
		try:
			W = eval(InputData.strAnalytPerturbationProfile)
		except:
			# error expression cannot be calculated
			W = 0

	else: # no Perturbation
		print('Unknown perturbation type. Use no Perturbation')
		if(InputData.Dimensionality==1):
			W = np.zeros(x.shape,float)
		elif(InputData.Dimensionality==2):
			W = np.zeros(x.shape,y.shape,float)

	#	print("<br>W=",W)
	return W,InputData

