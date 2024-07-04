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
from .operators import Get_Potential_Operator, Get_Potential_Operator2D,Get_Potential_Operator_from_file,Get_Potential_Operator_spinorbite,Get_Potential_Operator_orbital
from .observables import MidPointsOperator
from .CAP import Get_CAP_Magnetude


#*****************************************************************************#
# Type:
# 0 Plane waves basis set  without endpoint
# 1 Sine basis set
# 2 Laguerre polynomial
# 3 Hermite polynomial
# 4 Plane waves basis set with endpoint

def Get_Constant_Grid(N,x0,xL,Type=0):
	print("<br>Get_Constant_Grid Type=",Type)
	# build the grid point
	LBOX = xL-x0
	if(Type==0):
		#x			= np.linspace(x0,xL, num=N, endpoint=False)
		x			= np.linspace(x0,xL, num=N+1)
		delta_x0	= np.diff(x)
		delta_x		= np.concatenate([[delta_x0[0]+delta_x0[-1]],delta_x0[:-1]+delta_x0[1:]])/2
		x = x[:-1]

	elif(Type==1):
		x			= np.linspace(x0,xL, num=N+2)
		delta_x0	= np.diff(x)
		delta_x		= (delta_x0[:-1]+delta_x0[1:])/2
		x = x[1:-1] #x0 and xL are excluded from the grid

	else:
		print("<br>Uknown input method (use Type=0 for .. and Type=1 for..), Type=",Type)

	print('<br>   Get_Constant_Grid x.shape=',x.shape)
	print('<br>   Get_Constant_Grid delta_x0.shape=',delta_x0.shape)
	print('<br>   Get_Constant_Grid delta_x.shape=',delta_x.shape)
	return x,delta_x0,delta_x,LBOX

def Get_Constant_Grid2D(Nx,x0,xL,Ny,y0,yL,Type=0):
	x,delta_x0,delta_x,LBOXx = Get_Constant_Grid(Nx,x0,xL,Type)
	y,delta_y0,delta_y,LBOXy = Get_Constant_Grid(Ny,y0,yL,Type)

	return x,delta_x0,delta_x,y,delta_y0,delta_y,LBOXx,LBOXy

def Get_Grid(N,x0,xL=0,delta_x=None,Type=0):
	print("<br>Get_Grid Type=",Type)
	if(Type==0):
		x=np.zeros(N,float)   # declare of the grid point
		if delta_x==None:
			LBOX = xL-x0
			#x = np.linspace(x0,xL, num=N, endpoint=True)
			x = np.linspace(x0,xL, num=N+1)
			delta_x0 = np.diff(x)
		else:
			delta_x0 = delta_x
			x = x0+np.cumsum(delta_x[:-1])
			LBOX = x.max()-x0
		x = x[:-1]
		delta_x = np.concatenate([[delta_x0[0]+delta_x0[-1]],delta_x0[:-1]+delta_x0[1:]])/2
	if(Type==1):
		x=np.zeros(N,float)   # declare of the grid point
		if delta_x==None:
			LBOX = xL-x0
			x			= np.linspace(x0,xL, num=N+2)
			delta_x0 = np.diff(x)
		else:
			delta_x0 = delta_x
			x = x0+np.cumsum(delta_x[:-1])
			LBOX = x.max()-x0
		x = x[1:-1] #x0 and xL are excluded from the grid
		delta_x		= (delta_x0[:-1]+delta_x0[1:])/2
	elif(Type==2): #   Build x (or r) root of the Laguerre polynomial
		x,w = np.polynomial.laguerre.laggauss(N)
		#LN=np.zeros(N,float)
		#dLN=np.zeros(N,float)
		#for i in range(N):
			#LN[i],dLN[i] = Get_Laguerre_Polynomial_Derivative(N,x[i])
		#delta_x = exp(x)*x/((1-x/2)*LN+x*dLN)**2
		delta_x0 = np.diff(x)
		delta_x = MidPointsOperator(delta_x0)
		LBOX = x.max()-x.min()
	elif(Type==3): #   Build x root of the Hermite polynomial
		# x0 as center of well
		x,w = np.polynomial.hermite.hermgauss(N)
		delta_x0 = np.diff(x)
		delta_x = MidPointsOperator(delta_x0)
		x += x0
		LBOX = x.max()-x.min()
	else:
		print("<br>Uknown input method (use Type=0 for .. and Type=1 for..), Type=",Type)

	print('<br> Get_Grid x.shape=',x.shape)
	print('<br> Get_Grid delta_x0.shape=',delta_x0.shape)
	print('<br> Get_Grid delta_x.shape=',delta_x.shape)
	return x,delta_x0,delta_x,LBOX

def Get_Grid2D(Nx,x0,xL,delta_x,Ny,y0,yL,delta_y,Type=0):
	x,delta_x0,delta_x,LBOXx = Get_Grid(Nx,x0,xL,delta_x,Type)
	y,delta_y0,delta_y,LBOXy = Get_Grid(Ny,y0,yL,delta_y,Type)

	return x,delta_x0,delta_x,y,delta_y0,delta_y,LBOXx,LBOXy

def Get_Constant_K(Nk,delta_x):
	kmax = pi/delta_x.max()
	#k = np.linspace(-kmax,kmax+2*kmax/(Nk-1), num=Nk+1)
	#k = k[:-1]
	#delta_k=k[1]-k[0] 
	k = np.linspace(-kmax,kmax+kmax/Nk, num=2*Nk+1, endpoint=False)
	delta_k = MidPointsOperator(np.diff(k)) 

	return k,delta_k,kmax

def Get_Constant_K2D(Nkx,delta_x,Nky,delta_y):
	kx,delta_kx,kxmax = Get_Constant_K(Nkx,delta_x)
	ky,delta_ky,kymax = Get_Constant_K(Nky,delta_y)

	return kx,delta_kx,kxmax,ky,delta_ky,kymax

def Get_Laguerre_Polynomial_Derivative(N,x):
	if(N==0):
		LN=1e0; dLN=0e0

	else:
		LN=1e0-x; LNm1=1e0
		
		for i in range(2,N+1):
			LNm2=LNm1; LNm1=LN
			LN=((2*i-1-x)*LNm1-(i-1)*LNm2)/i

		dLN=N*(LN-LNm1)/x if x else -N*LN
		
	return LN,dLN


#******************************************************************************#
# Pot_Type:
# wells:
# 0 Infinite
# 1 Rectangular
# 2 Double wells
# 3 Double steps well
# 4 Analytic expression
# 5 parabolic (HO harmonic oscillator)
# 6 Delta potential
# 7 Double Delta potential
# 8 H2 (load from file)
# 9 H2+ (load from file)
# 10 load form file
# 11 Hydrogenoid Atoms
# 12 radioelement (alpha decay)
# 13 Morse
# 14 Amonnia molecule

# barriers:
# 20 --> free particle
# 21 --> step-well potential
# 22 --> barrier potential
# 23 --> delta-like potential
# 24 --> harmonic oscillator potential (barrier)
# 25 --> upload potential data file
# 30 --> free particle 2D
# 31 --> Infinite well for 1 slit
# 32 --> Infinite well for 2 slits
# 33 --> Infinite well for n slits does not work.. Need to fix potential with n_slits

# Get potential and check input data
def Get_Profile(InputData):
	OutData				= lambda:0
	if not hasattr(InputData, 'Dimensionality'):
		print ('<br>The Dimensionality of the task is not defined! Use default 1D (Dimensionality = 1)')
		InputData.Dimensionality	= 1
	print("<br>Dimensionality=",InputData.Dimensionality)


	# Type of kinetic operator discretisation
	if not hasattr(InputData, 'Type'):
		print ('<br>The type of discretization of the kinetic operator is not defined! Use default (Type = 0)')
		InputData.Type	= 0
	print("<br>Type=",InputData.Type)
	if(InputData.Type==0):
		print("<br>Using plane waves basis set without endpoint")
	elif(InputData.Type==4):
		print("<br>Using plane waves basis set with endpoint")
	elif(InputData.Type==1):
		print("<br>Using sine waves basis set without start and end points")
	elif(InputData.Type==3):
		print("<br>Using Hermite polynomiales basis set")
	elif(InputData.Type==2):
		print("<br>Using Laguerre polynomiales basis set")
	else:
		OutputData.Error = 'The type of discretization of the kinetic operator is not correct (Type = %i)!'%(InputData.Type)
		InputData.Type	= 0
		print ('<br>The type of discretization of the kinetic operator is not correct! Use default (Type = 0)')


	# complex profile
	if not hasattr(InputData, 'isCAP'):
		InputData.isCAP = 0
	print("<br>isCAP=",InputData.isCAP)

	# effective mass
	if not hasattr(InputData, 'mu'):
		InputData.mu		= 1
	if InputData.mu==0: # fictive particle reduced-mass
		me=1.
		mH=1.007825032/5.4857990946e-4
		m1=mH+me
		m2=m1
		InputData.mu=m1*m2/(m1+m2)
		print("<br>Defaul mass=%g u.a."%InputData.mu)
	print('<br>mu=',InputData.mu)


	# x-axis parameters
	if not hasattr(InputData, 'Nx'):
		InputData.Nx		= 100
	print("<br>Nx=",InputData.Nx)
	if not hasattr(InputData, 'x0'):
		InputData.x0		= 0
	print("<br>x0=",InputData.x0)
	if not hasattr(InputData, 'xL'):
		InputData.xL		= 1
	print("<br>xL=",InputData.xL)
	if not hasattr(InputData, 'xc'):
		InputData.xc		= (InputData.xL+InputData.x0)/2
	print("<br>xc=",InputData.xc)
	# y-axis parameters
	if InputData.Dimensionality==2:
		if not hasattr(InputData, 'Ny'):
			InputData.Ny		= 100
		print("<br>Ny=",InputData.Ny)
		if not hasattr(InputData, 'y0'):
			InputData.y0		= 0
		print("<br>y0=",InputData.y0)
		if not hasattr(InputData, 'yL'):
			InputData.yL		= 1
		print("<br>yL=",InputData.yL)
		if not hasattr(InputData, 'yc'):
			InputData.yc		= (InputData.yL+InputData.y0)/2
		print("<br>yc=",InputData.yc)


	if not hasattr(InputData, 'l'):
		InputData.l			= 0
	print("<br>l=",InputData.l)

	# Profile type
	if not hasattr(InputData, 'Pot_Type'):
		print ('<br>The potential profile is not define! Use default free (Pot_Type = 0)')
		InputData.Pot_Type	= 0
	print("<br>Pot_Type=",InputData.Pot_Type)


	if InputData.Pot_Type==0: # infinite well
		print("<br>infinite well")
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 500
		print("<br>V0=",InputData.V0)
		InputData.xmin_g,InputData.xmax_g	= InputData.x0-0.3*(InputData.xL-InputData.x0), InputData.xL+0.3*(InputData.xL-InputData.x0)
		
		if InputData.Type==0:
			x0, xL	= InputData.x0, ((InputData.Nx+1)*InputData.xL-InputData.x0)/InputData.Nx
		elif InputData.Type==1:
			x0, xL	= ((InputData.Nx+2)*InputData.x0-InputData.xL)/(InputData.Nx+1), ((InputData.Nx+2)*InputData.xL-InputData.x0)/(InputData.Nx+1)
		# Build the grid
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Constant_Grid(InputData.Nx,x0, xL,Type=InputData.Type)
		InputData.LBOXx = InputData.xL-InputData.x0

		# Get the potential energy operator
		OutData.V = Get_Potential_Operator(OutData.x,{'V0':InputData.V0},InputData.Pot_Type)

	elif InputData.Pot_Type==1: # rectangular well
		print("<br>rectangular well")
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 1
		print("<br>V0=",InputData.V0)
		# grid coordinates
		L_penetration = 0.7*(InputData.xL-InputData.x0)
		InputData.xmin_g,InputData.xmax_g		= InputData.x0-L_penetration, InputData.xL+L_penetration

		# Build the grid
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Constant_Grid(InputData.Nx,InputData.xmin_g, InputData.xmax_g,InputData.Type)

		# Get the potential energy operator
		OutData.V = Get_Potential_Operator(OutData.x,{'V0':InputData.V0,'x0':InputData.x0,'xL':InputData.xL},InputData.Pot_Type)

	elif (InputData.Pot_Type==2 or InputData.Pot_Type==3): # double or more complex wells
		print("<br>double or more complex wells")
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 1
		print("<br>V0=",InputData.V0)
		if not hasattr(InputData, 'V1'):
			InputData.V1			= InputData.V0
		print("<br>V1=",InputData.V1)
		V0 = np.array((InputData.V0,InputData.V1))
		if not hasattr(InputData, 'x0_1'):
			InputData.x0_1			= InputData.x0
		print("<br>x0_1=",InputData.x0_1)
		x0 = np.array((InputData.x0,InputData.x0_1))
		if not hasattr(InputData, 'xL_1'):
			InputData.xL_1			= InputData.xL
		print("<br>xL_1=",InputData.xL_1)
		xL = np.array((InputData.xL,InputData.xL_1))

		# grid coordinates
		InputData.xmin_g,InputData.xmax_g	= array((x0,xL)).min(), array((x0,xL)).max()
		L_penetration	= 0.5*(InputData.xmax_g-InputData.xmin_g)
		InputData.xmin_g,InputData.xmax_g	= InputData.xmin_g-L_penetration, InputData.xmax_g+L_penetration

		# Build the grid
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Constant_Grid(InputData.Nx,InputData.xmin_g, InputData.xmax_g,InputData.Type)

		# Get the potential energy operator
		OutData.V = Get_Potential_Operator(OutData.x,{'V0':V0,'x0':x0,'xL':xL},InputData.Pot_Type)

	elif InputData.Pot_Type==4: # Analytical profile of a well
		if not hasattr(InputData, 'strAnalytProfile'):
			InputData.strAnalytProfile			= 'x**2'
		print("<br>strAnalytProfile=",InputData.strAnalytProfile)

		print("<br>Analytical profile of a well")
		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL

		# Build the grid
		x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Constant_Grid(InputData.Nx, InputData.x0,InputData.xL,InputData.Type)

		# Get the potential energy operator
		try:
			OutData.V = (eval(InputData.strAnalytProfile))
		except:
			OutData.Error = 'Analytical expression is not correct! Check expression.'
			return OutData,InputData
		OutData.x = x

	elif InputData.Pot_Type==5 or InputData.Pot_Type==24: # Harmonic Oscillator Potential
		if not hasattr(InputData, 'omegax'):
			InputData.omegax			= 1
		print("<br>omegax=",InputData.omegax)


		if InputData.Dimensionality==1:
			if InputData.Type==3:
				OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Grid(InputData.Nx,InputData.xc,Type=InputData.Type)
				InputData.x0,InputData.xL = OutData.x.min(),OutData.x.max()
			else: # InputData.Type==0:
				print("<br>Type=",InputData.Type)
				OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Constant_Grid(InputData.Nx,InputData.x0,InputData.xL,InputData.Type)

			#   Get the potential energy operator
			OutData.V = Get_Potential_Operator(OutData.x,{'V0':0.5*(InputData.omegax)**2*InputData.mu,'xc':InputData.xc},Pot_Type=5)
		elif InputData.Dimensionality==2:
			if InputData.Type==3:
				OutData.x,InputData.delta_x0,InputData.delta_x,OutData.y,InputData.delta_y0,InputData.delta_y,InputData.LBOXx,InputData.LBOXy = Get_Grid2D(InputData.Nx,InputData.xc,None,None,InputData.Ny,InputData.yc,None,None,Type=InputData.Type)
				InputData.x0,InputData.xL = OutData.x.min(),OutData.x.max()
				InputData.y0,InputData.yL = OutData.y.min(),OutData.y.max()
			else: # InputData.Type==0:
				OutData.x,InputData.delta_x0,InputData.delta_x,OutData.y,InputData.delta_y0,InputData.delta_y,InputData.LBOXx,InputData.LBOXy = Get_Constant_Grid2D(InputData.Nx,InputData.x0,InputData.xL,InputData.Ny,InputData.y0,InputData.yL,InputData.Type)
			InputData.ymin_g,InputData.ymax_g	= InputData.y0,InputData.yL

			if not hasattr(InputData, 'omegay'):
				InputData.omegay			= 1
			print("<br>omegay=",InputData.omegay)
			if InputData.omegax==InputData.omegay:
				InputData.omegay += 1e-6

			OutData.V = Get_Potential_Operator2D(OutData.x,OutData.y,{'V0x':0.5*(InputData.omegax)**2*InputData.mu,'V0y':0.5*(InputData.omegay)**2*InputData.mu,'xc':InputData.xc,'yc':InputData.yc},Pot_Type=5)
		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL

	elif InputData.Pot_Type==6: # Delta potential
		print("<br>Delta potential")
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 1
		print("<br>V0=",InputData.V0)
		if not hasattr(InputData, 'sigma'):
			InputData.sigma			= 1
		print("<br>sigma=",InputData.sigma)

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL

		# Build the grid
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Constant_Grid(InputData.Nx, InputData.x0,InputData.xL,InputData.Type)

		# Get the potential energy operator
#		InputData.xc = OutData.x[int(InputData.Nx/2.)]
		OutData.V = Get_Potential_Operator(OutData.x,{'V0':InputData.V0,'xc':InputData.xc,'sigma':InputData.sigma},InputData.Pot_Type)

	elif InputData.Pot_Type==7: # double Delta potential
		print("<br>double Delta potential")
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 1
		print("<br>V0=",InputData.V0)
		if not hasattr(InputData, 'sigma'):
			InputData.sigma			= 1
		print("<br>sigma=",InputData.sigma)
		if not hasattr(InputData, 'd'):
			InputData.d			= 1
		print("<br>d=",InputData.d)
		Nanalytic	= 20

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL

		# Build the grid
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Constant_Grid(InputData.Nx, InputData.x0,InputData.xL,InputData.Type)

		# Get the potential energy operator
		OutData.V = Get_Potential_Operator(OutData.x,{'V0':InputData.V0,'xc':np.array([InputData.xc-InputData.d/2,InputData.xc+InputData.d/2]),'sigma':InputData.sigma},InputData.Pot_Type)

	elif ((InputData.Pot_Type==8) or (InputData.Pot_Type==9)): # load from file
		if (InputData.Pot_Type==8):
			fname = 'H2.dat'
		elif (InputData.Pot_Type==9):
			fname = 'H2+.dat'

		# Get the potential energy operator
		OutData.V,OutData.x,InputData.x0,InputData.xL,InputData.delta_x0 = Get_Potential_Operator_from_file(fname)
		if (InputData.Type==0):
			InputData.delta_x0	= np.concatenate([InputData.delta_x0,[InputData.delta_x0[-1]]])
			InputData.delta_x	= np.concatenate([[InputData.delta_x0[0]+InputData.delta_x0[-1]],InputData.delta_x0[:-1]+InputData.delta_x0[1:]])/2
			InputData.LBOXx = (InputData.xL+InputData.delta_x0[-1])-InputData.x0
		elif (InputData.Type==1):
			InputData.delta_x0	= np.concatenate([[InputData.delta_x0[0]],InputData.delta_x0,[InputData.delta_x0[-1]]])
			InputData.delta_x = (InputData.delta_x0[:-1]+InputData.delta_x0[1:])/2
			InputData.LBOXx = (InputData.xL+InputData.delta_x0[-1])-(InputData.x0-InputData.delta_x0[0])

		InputData.Nx = len(OutData.x)
		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL

	elif (InputData.Pot_Type==10): # get external data
		OutData.x = np.array(InputData.x)
		OutData.V = np.array(InputData.V)
		InputData.delta_x0 = np.diff(OutData.x)

		if (InputData.Type==0):
			InputData.delta_x0	= np.concatenate([InputData.delta_x0,[InputData.delta_x0[-1]]])
			InputData.delta_x	= np.concatenate([[InputData.delta_x0[0]+InputData.delta_x0[-1]],InputData.delta_x0[:-1]+InputData.delta_x0[1:]])/2
			InputData.LBOXx = (InputData.xL+InputData.delta_x0[-1])-InputData.x0
		elif (InputData.Type==1):
			InputData.delta_x0	= np.concatenate([[InputData.delta_x0[0]],InputData.delta_x0,[InputData.delta_x0[-1]]])
			InputData.delta_x = (InputData.delta_x0[:-1]+InputData.delta_x0[1:])/2
			InputData.LBOXx = (InputData.xL+InputData.delta_x0[-1])-(InputData.x0-InputData.delta_x0[0])

		InputData.Nx = len(OutData.x)
		InputData.xmin_g,InputData.xmax_g	= OutData.x.min(),OutData.x.max()

	elif (InputData.Pot_Type==11): # Hydrogenoid Atoms
		print ('<br>Hydrogenoid Atoms')
		if InputData.x0<=0:
			InputData.x0 = 1e-8
		print("<br>x0=",InputData.x0)
		if not hasattr(InputData, 'Z'):
			InputData.Z			= 1 #  for Hydrogen atom
		print("<br>Z=",InputData.Z)
		if not hasattr(InputData, 'De'):
			InputData.De			= 0.04730 # 0.04764
		print("<br>De=",InputData.De)
		if not hasattr(InputData, 'E_I'):
			InputData.E_I			= 0.5
		print("<br>E_I=",InputData.E_I)

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL

		#   Build the grid
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Grid(InputData.Nx,InputData.xmin_g,InputData.xmax_g,Type=InputData.Type)
		InputData.x0,InputData.xL = OutData.x[0],60 # OutData.x[-1]

		OutData.V = Get_Potential_Operator(OutData.x,{'Z':InputData.Z},InputData.Pot_Type)
		if(InputData.l==0):
			OutData.V[0]=1e5      # to enssure the simualtion box for l=0
			#OutData.V[-1]=1e5

	elif (InputData.Pot_Type==12): # alpha decay
		print ('<br>alpha decay')
		if not hasattr(InputData, 'LCAP'):
			InputData.LCAP		= 30
		print("<br>LCAP=",InputData.LCAP)
		if not hasattr(InputData, 'V0'):
			InputData.V0		= -115
		print("<br>V0=",InputData.V0)
		if not hasattr(InputData, 'V1'):
			InputData.V1			= 500
		print("<br>V1=",InputData.V1)
		if not hasattr(InputData, 'Z'):
			InputData.Z			= 84
		print("<br>Z=",InputData.Z)
		if not hasattr(InputData, 'delta'):
			InputData.delta			= 1
		print("<br>delta=",InputData.delta)
		if not hasattr(InputData, 'sigma'):
			InputData.sigma			= 0.8
		print("<br>sigma=",InputData.sigma)
		if not hasattr(InputData, 'xL'):
			InputData.xL			= 200
		print("<br>xL=",InputData.xL)
		if not hasattr(InputData, 'A'):
			InputData.A			= 212
		print("<br>A=",InputData.A)
		InputData.R = 1.07*(4**(1./3.)+(InputData.A-4)**(1./3.));
		print("<br>R=",InputData.R) # Radius of alpha particle and daughter nucleus in fm


		if not hasattr(InputData, 'n'):
			InputData.n			= 5
		print("<br>n=",InputData.n)
		n		= InputData.n # the number of rectangular sub-barriers


		# convertion to a.u.
		InputData.x0		= InputData.x0/au_to_fm
		InputData.xL		= InputData.xL/au_to_fm
		InputData.R			= InputData.R/au_to_fm
		InputData.sigma		= InputData.sigma/au_to_fm
		InputData.LCAP		= InputData.LCAP/au_to_fm 
		InputData.mu		= InputData.mu/au_to_amu
		InputData.V0		= InputData.V0/au_to_MeV
		InputData.V1		= InputData.V1/au_to_MeV
		InputData.Ekin_min	= InputData.Ekin_min/au_to_MeV
		InputData.Ekin_max	= None
		#Ekin_max = Ekin_max/au_to_MeV

		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Constant_Grid(InputData.Nx,InputData.x0,InputData.xL,InputData.Type)
		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL

		# Complex absorbing potential part
		if(InputData.isCAP):
			InputData.A2,Ekin_average,lambda_dB_average = Get_CAP_Magnetude(InputData.Ekin_min,InputData.Ekin_max,InputData.mu,InputData.LCAP)
		else:
			InputData.A2 = 0

		print("<br>mass of alpha particle=%g a.u."%InputData.mu)
		print("<br>radius of alpha particle (Nuclear radius) R=%g fm"%(InputData.R*au_to_fm))
		if(InputData.isCAP):
			print('<br><Ekin>=%g MeV  lambda_dB=%g fm ' %(Ekin_average*au_to_MeV,lambda_dB_average*au_to_fm))
			print('<br>L/lambda_dB=%g delta r=%g fm'%(InputData.LCAP/lambda_dB_average,InputData.delta_x[0]*au_to_fm))
		print('<br>The optimal A2 is %g in MeV and %g in a.u.' %(InputData.A2*au_to_MeV, InputData.A2))
		print("<br>V0=%g MeV"%(InputData.V0*au_to_MeV))
		print("<br>V1=%g MeV"%(InputData.V1*au_to_MeV))


		OutData.V,OutData.V_Wall,OutData.V_WS,OutData.V_LR,OutData.V_SR,OutData.minloc,OutData.maxloc,OutData.minval,OutData.maxval,OutData.xmax,OutData.xmin,OutData.V_Model = Get_Potential_Operator(OutData.x,{'V0':InputData.V0,'V1':InputData.V1,'R':InputData.R,'Z':InputData.Z,'delta':InputData.delta,'sigma':InputData.sigma,'LCAP':InputData.LCAP,'xL':InputData.xL,'A2':InputData.A2},InputData.Pot_Type)


	elif(InputData.Pot_Type==13): # Morse potential for simulating diotomics
		print ('<br>Morse potential for simulating diotomics')
		if not hasattr(InputData, 'xc0'):
			InputData.xc0		= 1
		print("<br>xc0=",InputData.xc0)
		if not hasattr(InputData, 'alpha'):
			InputData.alpha			= 1.3696
		print("<br>alpha=",InputData.alpha)
		if not hasattr(InputData, 'De'):
			InputData.De			= 0.04730 # 0.04764
		print("<br>De=",InputData.De)
		if InputData.x0<=0:
			InputData.x0 = 1e-8
		print("<br>x0=",InputData.x0)

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Grid(InputData.Nx,InputData.xmin_g,InputData.xmax_g,Type=InputData.Type)

		OutData.V = Get_Potential_Operator(OutData.x,{'De':InputData.De,'xc':InputData.xc0,'alpha':InputData.alpha},InputData.Pot_Type)

	elif(InputData.Pot_Type==14):  # Ammonia molecule
		print ('<br>Ammonia molecule')
		if not hasattr(InputData, 'omega'):
			InputData.omega = 0.004   # frequency
		print ('<br>omega=',InputData.omega)
		if not hasattr(InputData, 'xc0'):
			InputData.xc0		= 1
		print("<br>xc0=",InputData.xc0)
		if not hasattr(InputData, 'c'):
			InputData.c			= 0.05684
		print("<br>c=",InputData.c)
		if not hasattr(InputData, 'alpha'):
			InputData.alpha			= 1.3696
		print("<br>alpha=",InputData.alpha)
		if not hasattr(InputData, 'De'):
			InputData.De			= 0.04730 # 0.04764
		print("<br>De=",InputData.De)
#		if InputData.x0<=0:
#			InputData.x0 = 1e-8
#		print("<br>x0=",InputData.x0)

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Grid(InputData.Nx,InputData.xmin_g,InputData.xmax_g,Type=InputData.Type)

		OutData.V = Get_Potential_Operator(OutData.x,{'V0':1/2*InputData.omega**2*InputData.mu,'De':InputData.De,'xc':InputData.xc0,'alpha':InputData.alpha,'c':InputData.c},InputData.Pot_Type)


	elif(InputData.Pot_Type==20):
		print ('<br>free particle')

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL
		#free-particle 2D
		if InputData.Dimensionality==1:
			OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Grid(InputData.Nx,InputData.xmin_g,InputData.xmax_g,Type=InputData.Type)

			OutData.V = Get_Potential_Operator(OutData.x,None,InputData.Pot_Type)
		elif InputData.Dimensionality==2:
			InputData.ymin_g,InputData.ymax_g	= InputData.y0,InputData.yL

			OutData.x,InputData.delta_x0,InputData.delta_x,OutData.y,InputData.delta_y0,InputData.delta_y,InputData.LBOXx,InputData.LBOXy	= Get_Constant_Grid2D(InputData.Nx,InputData.x0,InputData.xL,InputData.Ny,InputData.y0,InputData.yL,InputData.Type)
			
			#   Build the potential 
			OutData.V = Get_Potential_Operator2D(OutData.x,OutData.y,None,InputData.Pot_Type)

	elif(InputData.Pot_Type==21):
		print ('<br>step-well potential')
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 1
		print("<br>V0=",InputData.V0)
		if not hasattr(InputData, 'dx'):
			InputData.dx			= 1e-2
		print("<br>dx=",InputData.dx)

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Grid(InputData.Nx,InputData.xmin_g,InputData.xmax_g,Type=InputData.Type)

		OutData.V = Get_Potential_Operator(OutData.x,{'V0':InputData.V0,'xc':InputData.xc,'dx':InputData.dx},InputData.Pot_Type)

	elif(InputData.Pot_Type==22): #barrier potential
		print ('<br>barrier potential')
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 1
		print("<br>V0=",InputData.V0)
		if not hasattr(InputData, 'd'):
			InputData.d			= 1
		print("<br>d=",InputData.d)
		if not hasattr(InputData, 'dx'):
			InputData.dx			= 1e-2
		print("<br>dx=",InputData.dx)

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Grid(InputData.Nx,InputData.xmin_g,InputData.xmax_g,Type=InputData.Type)

		OutData.V = Get_Potential_Operator(OutData.x,{'V0':InputData.V0,'xc':InputData.xc,'d':InputData.d,'dx':InputData.dx},InputData.Pot_Type)

	elif(InputData.Pot_Type==23): #delta-like potential
		print ('<br>delta-like potential')
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 1
		print("<br>V0=",InputData.V0)
		if not hasattr(InputData, 'sigma'):
			InputData.sigma			= 1
		print("<br>sigma=",InputData.sigma)

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL
		OutData.x,InputData.delta_x0,InputData.delta_x,InputData.LBOXx = Get_Grid(InputData.Nx,InputData.xmin_g,InputData.xmax_g,Type=InputData.Type)

		OutData.V = Get_Potential_Operator(OutData.x,{'V0':InputData.V0,'xc':InputData.xc,'sigma':InputData.sigma},InputData.Pot_Type)


	elif(InputData.Pot_Type==31): # slits
		if InputData.Dimensionality==1:
			print ('<br>The Dimensionality of the task is changed to 2D (Dimensionality = 2)')
			InputData.Dimensionality = 2
		#   Barrier parameters
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 100
		print("<br>V0=",InputData.V0)
		if not hasattr(InputData, 'slit_d'):
			InputData.slit_d		= 5
		print("<br>slit_d=",InputData.slit_d)
		if not hasattr(InputData, 'slit_ls'):
			InputData.slit_ls		= InputData.d
		print("<br>slit size, slit_ls=",InputData.slit_ls)

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL
		InputData.ymin_g,InputData.ymax_g	= InputData.y0,InputData.yL

		OutData.x,InputData.delta_x0,InputData.delta_x,OutData.y,InputData.delta_y0,InputData.delta_y,InputData.LBOXx,InputData.LBOXy	= Get_Constant_Grid2D(InputData.Nx,InputData.x0,InputData.xL,InputData.Ny,InputData.y0,InputData.yL,InputData.Type)

		OutData.V = Get_Potential_Operator2D(OutData.x,OutData.y,{'V0':InputData.V0,'l':InputData.slit_d,'ls':InputData.slit_ls,'xc':InputData.xc,'yc':InputData.yc},InputData.Pot_Type)

	elif(InputData.Pot_Type==32):
		if InputData.Dimensionality==1:
			print ('<br>The Dimensionality of the task is changed to 2D (Dimensionality = 2)')
			InputData.Dimensionality = 2
		#   Barrier parameters
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 100
		print("<br>V0=",InputData.V0)
		if not hasattr(InputData, 'slit_d'):
			InputData.slit_d		= 8
		print("<br>slit_d=",InputData.slit_d)
		if not hasattr(InputData, 'slit_ls'):
			InputData.slit_ls		= InputData.slit_d
		print("<br>slit size, slit_ls=",InputData.slit_ls)
		if not hasattr(InputData, 'slit_ly'):
			InputData.slit_ly		= InputData.slit_ls
		print("<br>distance between 2 slits, slit_ly=",InputData.slit_ly)

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL
		InputData.ymin_g,InputData.ymax_g	= InputData.y0,InputData.yL

		OutData.x,InputData.delta_x0,InputData.delta_x,OutData.y,InputData.delta_y0,InputData.delta_y,InputData.LBOXx,InputData.LBOXy	= Get_Constant_Grid2D(InputData.Nx,InputData.x0,InputData.xL,InputData.Ny,InputData.y0,InputData.yL,InputData.Type)

		OutData.V = Get_Potential_Operator2D(OutData.x,OutData.y,{'V0':InputData.V0,'l':InputData.slit_d,'ls':InputData.slit_ls,'ly':InputData.slit_ly,'xc':InputData.xc,'yc':InputData.yc},InputData.Pot_Type)

	elif(InputData.Pot_Type==33):
		if InputData.Dimensionality==1:
			print ('<br>The Dimensionality of the task is changed to 2D (Dimensionality = 2)')
			InputData.Dimensionality = 2
		#   Barrier parameters
		if not hasattr(InputData, 'V0'):
			InputData.V0		= 100
		print("<br>V0=",InputData.V0)
		if not hasattr(InputData, 'slit_d'):
			InputData.slit_d		= 8
		print("<br>slit_d=",InputData.slit_d)
		if not hasattr(InputData, 'slit_ls'):
			InputData.slit_ls		= InputData.slit_d
		print("<br>slit size, slit_ls=",InputData.slit_ls)
		if not hasattr(InputData, 'slit_ly'):
			InputData.slit_ly		= InputData.slit_ls
		print("<br>distance between 2 slits, slit_ly=",InputData.slit_ly)
		if not hasattr(InputData, 'slit_n'):
			InputData.slit_n		= 3
		print("<br>number of slits=",InputData.slit_n)

		if not hasattr(InputData, 'n_slits'):
			InputData.n_slits		= 3
		print("<br>number of slits, n_slits=",InputData.n_slits)
		if(InputData.n_slits*InputData.slit_ls+(InputData.n_slits-1+2)*InputData.slit_ly>=(InputData.yL-InputData.y0)):
			sys.exit('Number of slits too large for the current grid!!!') 

		InputData.xmin_g,InputData.xmax_g	= InputData.x0,InputData.xL
		InputData.ymin_g,InputData.ymax_g	= InputData.y0,InputData.yL

		OutData.x,InputData.delta_x0,InputData.delta_x,OutData.y,InputData.delta_y0,InputData.delta_y,InputData.LBOXx,InputData.LBOXy	= Get_Constant_Grid2D(InputData.Nx,InputData.x0,InputData.xL,InputData.Ny,InputData.y0,InputData.yL,InputData.Type)

		OutData.V = Get_Potential_Operator2D(OutData.x,OutData.y,{'V0':InputData.V0,'l':InputData.slit_d,'ls':InputData.slit_ls,'ly':InputData.slit_ly,'xc':InputData.xc,'yc':InputData.yc},InputData.Pot_Type)

	else:
		OutputData.Error = 'The type of potential energy profile is not correct (Pot_Type = %i)!'%(InputData.Pot_Type)

	print("<br>delta_x0.shape=",InputData.delta_x0.shape)
	print("<br>delta_x.shape=",InputData.delta_x.shape)
	print("<br>x.shape=",OutData.x.shape)


	if not InputData.l==0:
		print('<br>l is not 0: add to the potential energy')
#		print("<br>0 V=",OutData.V)
		if 	InputData.Dimensionality==1:
			OutData.V += Get_Potential_Operator_orbital(OutData.x,InputData.l,InputData.mu)
#		print("<br>1 V=",OutData.V)
	if not hasattr(InputData, 's'):
		InputData.s = None
	print('<br>s=',InputData.s)
	if not hasattr(InputData, 'J'):
		InputData.J = None
	print('<br>J=',InputData.J)
	if (InputData.s is None) or (InputData.J is None):
		print('<br>s=noneType or J=noneType: Calculation without spin-orbit interaction...')
	else:
		print('<br>Total momentum and spin are defined: add the spin-orbit interaction to the potential energy')
#		print("<br>0 V=",OutData.V)
		OutData.V += Get_Potential_Operator_spinorbite(OutData.x,InputData.s,InputData.l,InputData.J,1,1)
#		print("<br>1 V=",OutData.V)

	print('<br>xmin_g=',InputData.xmin_g)
	print('<br>xmax_g=',InputData.xmax_g)

	return OutData,InputData



