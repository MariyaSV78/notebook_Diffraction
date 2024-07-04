
from __future__ import division
from math import *
import numpy as np
from pylab import *
from numpy import linalg as LA
from numpy.polynomial.hermite import hermval
import scipy.optimize as op
import scipy.special

from .observables import MidPointsOperator
from .constants import ci


#*****************************************************************************#
def Get_Analytical_Solutions(N,x,InputData,Pot_Type):

	if Pot_Type==0: # inf
		mu		= InputData['mu']		if 'mu'		in InputData	else 1
		L		= InputData['L']		if 'L'		in InputData	else 1
		k0 = pi/L*(np.arange(N)+1) # here n+1 to avoid n=0
		E_analytic = k0**2/(2*mu)
		xx = array([x]).T-x[0]
		psi_analytic = sqrt(2/L)*sin((xx[...,:]*k0).T)
	elif Pot_Type==1:
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		E		= InputData['E']		if 'E'		in InputData	else 1
		mu		= InputData['mu']		if 'mu'		in InputData	else 1
		L		= InputData['L']		if 'L'		in InputData	else 1
		epsilon	= InputData['epsilon']	if 'epsilon'	in InputData	else 1
		Nanalytic = N

		L=L/2
		z0=L*sqrt(2*mu*V0)
		print('<br>z0=',z0)

		z=linspace(0.,z0,Nanalytic)
		Eanalytic  = np.zeros(Nanalytic,float)
		z_analytic = np.zeros(Nanalytic,float)

		#h=tan(z)
		#h[abs(h)>100] = 100

		ns=0
		for i in range(Nanalytic-1):
			z1 = z[i]
			z2 = z[i+1]
			#e1=(z1/L)**2/(2*mu)-V0
			#e2=(z2/L)**2/(2*mu)-V0
			#if(i%2==0 and e1<0 and e2<0 and f1_2(z1,z0)*f1_2(z2,z0)<0. and E[ns]<0):
			if(f1_2(z1,z0)*f1_2(z2,z0)<0):
				#tmp_z=op.newton(f1,z1,df1, args=(), tol=1.48e-03, maxiter=100)
				#tmp_z=op.bisect(f1,z1,z2)
				tmp_z = op.fsolve(f1_2,z1,args=(z0,))
				tmp_E = (tmp_z/L)**2/(2*mu)-V0
				if(abs(f1_2(tmp_z,z0))<epsilon):
					Eanalytic[ns]  = tmp_E
					z_analytic[ns] = tmp_z
					print('<br>paire',i,tmp_z,f1_2(tmp_z,z0))
					ns=ns+1
			#elif((not i%2==0) and e1 <0 and e2<0 and f2_2(z1,z0)*f2_2(z2,z0)<0. and E[ns]<0):
			elif(f2_2(z1,z0)*f2_2(z2,z0)<0):
				#tmp_z=op.newton(f2,z1,df2, args=(), tol=1.48e-03, maxiter=100)
				#tmp_z=op.bisect(f2,z1,z2)
				tmp_z = op.fsolve(f2_2,z1,args=(z0,))
				tmp_E = (tmp_z/L)**2/(2*mu)-V0
				if(abs(f2_2(tmp_z,z0))<epsilon):
					Eanalytic[ns]  = tmp_E
					z_analytic[ns] = tmp_z
					print('<br>impaire',i,tmp_z,f2_2(tmp_z,z0))
					ns=ns+1
	
		return Eanalytic[:ns],z_analytic[:ns],ns

	elif (Pot_Type==5):
		mu		= InputData['mu']		if 'mu'		in InputData	else 1
		omega	= InputData['omega']	if 'omega'	in InputData	else 1

		n = np.arange(N)
		E_analytic = (n+0.5)*omega
	
		alpha = sqrt(mu*omega)
	
		psi_analytic = np.zeros((N, x.size),float)
		Hn_coef = np.zeros(x.size,float)
	
		for n in range(min(13,N)):
			beta = 1/sqrt(2**n*factorial(n))*(alpha**2/pi)**(1/4)
			Hn_coef[n] = 1
			psi_analytic[n, :]=beta*exp(-(alpha*x)**2/2)*hermval(alpha*x, Hn_coef)
			Hn_coef[n] = 0

	elif (Pot_Type==6):
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		mu		= InputData['mu']		if 'mu'		in InputData	else 1
		xc		= InputData['xc']		if 'xc'		in InputData	else 0
		lambda_0	= InputData['lambda_0']	if 'lambda_0'	in InputData	else 1

		Nanalytic = N
#		E_analytic,ns = Get_Analytical_Solutions_delta(len(x)x,mu,V0,lambda_0,d,Nanalytic)
		E_analytic = np.array(-1./(2*mu*lambda_0**2)).reshape(1)
		psi_analytic = (sqrt(1/lambda_0)*exp(-abs(x-xc)/lambda_0)).reshape((1,len(x)))

	elif (Pot_Type==7):
		V0		= InputData['V0']		if 'V0'		in InputData	else 1
		mu		= InputData['mu']		if 'mu'		in InputData	else 1
		xc		= InputData['xc']		if 'xc'		in InputData	else 0
		d		= InputData['d']		if 'd'		in InputData	else 0
		lambda_0	= InputData['lambda_0']	if 'lambda_0'	in InputData	else 1

		Nanalytic = N
		E_analytic,ns = Get_Analytical_Solutions_Ndelta(len(x),x,mu,V0,lambda_0,d,Nanalytic)
		E_analytic = E_analytic[:2]
		_,psi1 = Get_Analytical_Solutions(Nanalytic,x,{'mu':mu,'lambda_0':lambda_0,'xc':xc-d/2},Pot_Type=6)
		_,psi2 = Get_Analytical_Solutions(Nanalytic,x,{'mu':mu,'lambda_0':lambda_0,'xc':xc+d/2},Pot_Type=6)
		psi_analytic = (np.vstack((psi1+psi2,psi1-psi2))/sqrt(2))

	elif (Pot_Type==11):
		E_I		= InputData['E_I']		if 'E_I'		in InputData	else 1
		mu		= InputData['mu']		if 'mu'		in InputData	else 1
		Z		= InputData['Z']		if 'Z'		in InputData	else 0
		l		= InputData['l']		if 'l'		in InputData	else 0

		Neig = N
		N=len(x)
		Eanalytic=np.zeros(Neig,float) 
		Rnl=np.zeros((N,Neig),float)
	
		a0=1/Z
		for k in range(Neig):
			n=k+l+1
			Eanalytic[k]=-E_I*Z**2/n**2
			coef_num=factorial(n-l-1)
			coef_denom=2*n*(factorial(n+l))**3
			coef=sqrt((2./(n*a0))**3*coef_num/coef_denom)
			
			for ir in range(N):
				r_ir=x[ir]/(n*a0)
				#Lnl=scipy.special.eval_genlaguerre(n-l-1,2*l+1,2*r_ir)
				Lnl=scipy.special.assoc_laguerre(2*r_ir,n-l-1,2*l+1)
				Lnl=Lnl*factorial(n-l-1+2*l+1) # to ensure the normalization 
				Rnl[ir,k]=coef*exp(-r_ir)*(2*r_ir)**l*Lnl
	
		return Eanalytic,Rnl


	return E_analytic,psi_analytic


def Get_Analytical_Solutions_delta(N,x,mu,V0,lambda_0,N_analytic):
	epsilon = 1e-5
	ns = 0
	e = np.linspace(V0,0.,N_analytic)
	E_analytic = np.zeros(N_analytic,float)

	for i in range(N_analytic-1):
		a = e[i]
		b = e[i+1]

		E_analytic[ns] = Get_Zero_Dichotomique(a,b,lambda_0,epsilon,0)
		E_analytic[ns] = Get_Zero_Dichotomique(a,b,lambda_0,epsilon,1)
		ns = ns+1

	return E_analytic,ns


def Get_Analytical_Solutions_Ndelta(N,x,mu,V0,lambda_0,d,N_analytic):
	epsilon=1e-2
	ns=0
	q=np.linspace(0.,10.,N_analytic)
	E_analytic = np.zeros(N,float)

	q2=0.5

	tmp_q0=op.newton(f1,q2,df1, args=(lambda_0,d), tol=1.48e-03, maxiter=100)
	tmp_E=-tmp_q0**2/(2*mu)

	if(abs(f1(tmp_q0,lambda_0,d))<epsilon):
		E_analytic[ns]=tmp_E
#		print('<br>tmp_q0=',tmp_q0,'f1=',f1(tmp_q0,lambda_0,d))
		ns=ns+1


	tmp_q1=op.newton(f2,q2,df2, args=(lambda_0,d), tol=1.48e-03, maxiter=100)
	tmp_E=-tmp_q1**2/(2*mu)

	if(abs(f2(tmp_q1,lambda_0,d))<epsilon):
		E_analytic[ns]=tmp_E
#		print('<br>tmp_q1=',tmp_q1,'f2=',f2(tmp_q1,lambda_0,d))
		ns=ns+1

	return E_analytic,ns

#*****************************************************************************#
def Get_Zero_Dichotomique(a,b,lambda_0,epsilon,state):
	g,d=a,b
	if(state==0):
		while (d-g) > epsilon:
			m=(g+d)/2
			if f1(g,lambda_0,d)*f1(m,lambda_0,d)<0:
				d=m #m sup a la racine
			elif f1(g,lambda_0,d)*f1(m,lambda_0,d)>0: #m inf a la racine
				g=m
			else: #coup de chance f(m,lambda_0,d)=0
				return m
	elif(state==1):
		while (d-g) > epsilon:
			m=(g+d)/2
			if f2(g,lambda_0,d)*f2(m,lambda_0,d)<0:
				d=m #m sup a la racine
			elif f2(g,lambda_0,d)*f2(m,lambda_0,d)>0: #m inf a la racine
				g=m
			else: #coup de chance f(m,lambda_0,d)=0
				return m
	return (g+d)/2

def f1(z,lambda_0,d):
	return z*lambda_0-1-exp(-z*d)

def f2(z,lambda_0,d):
	return z*lambda_0-1+exp(-z*d)

def df1(z,lambda_0,d):
	return lambda_0+d*exp(-z*d)

def df2(z,lambda_0,d):
	return lambda_0-d*exp(-z*d)

#**square***************************************************************************#
def f1_2(z,z0):
	#return tan(z)-sqrt((z0/z)**2-1)
	return sin(z)*z-cos(z)*sqrt(z0**2-z**2)
def f2_2(z,z0):
	#return -1./tan(z)-sqrt((z0/z)**2-1)
	return -cos(z)*z-sin(z)*sqrt(z0**2-z**2)
def df1_2(z,z0):
	return 1./cos(z)**2+z0**2/(z**3*sqrt((z0/z)**2-1))
def df2_2(z,z0):
	return -1./sin(z)**2-z0**2/(z**3*sqrt((z0/z)**2-1))


def Get_Analytical_Solutions_square0(L,mu,V0,Nanalytic,epsilon=1e-3):
	z0 = L*sqrt(2*mu*V0)
	print('z0=',z0)

	z			= linspace(0.,z0,Nanalytic)
	Eanalytic	= zeros(N,float)
	z_analytic	= zeros(N,float)

	h=tan(z)
	h[abs(h)>100] = 100

	ns = 0
	for i in range(Nanalytic-1):
		z1=z[i]
		z2=z[i+1]
		e1=(z1/L)**2/(2*mu)-V0
		e2=(z2/L)**2/(2*mu)-V0
#		if(i%2==0 and e1 <0. and e2<0. and f1_2(z1,z0)*f1_2(z2,z0)<0. and E[ns]<0):
		if(e1 <0. and e2<0. and f1_2(z1,z0)*f1_2(z2,z0)<0. and E[ns]<0):
			#tmp_z=op.newton(f1,z1,df1, args=(), tol=1.48e-03, maxiter=100)
			#tmp_z=op.bisect(f1,z1,z2)
			tmp_z=op.fsolve(f1_2,z1,args=(z0))
			tmp_E=(tmp_z/L)**2/(2*mu)-V0
			if(abs(f1_2(tmp_z,z0))<epsilon):
				Eanalytic[ns]=tmp_E
				z_analytic[ns]=tmp_z
				print('<br>paire',i,tmp_z,f1(tmp_z))
				ns=ns+1
#		elif((not i%2==0) and e1 <0. and e2<0. and f2_2(z1,z0)*f2_2(z2,z0)<0. and E[ns]<0):
		elif(e1 <0. and e2<0. and f2_2(z1,z0)*f2_2(z2,z0)<0. and E[ns]<0):
			#tmp_z=op.newton(f2,z1,df2, args=(), tol=1.48e-03, maxiter=100)
			#tmp_z=op.bisect(f2,z1,z2)
			tmp_z=op.fsolve(f2_2,z1,args=(z0))
			tmp_E=(tmp_z/L)**2/(2*mu)-V0
			if(abs(f2_2(tmp_z,z0))<epsilon):
				Eanalytic[ns]=tmp_E
				z_analytic[ns]=tmp_z
				print('<br>impaire',i,tmp_z,f2(tmp_z))
				ns=ns+1

	return Eanalytic[:ns-1],z_analytic[:ns-1],ns

#*******************************temporal**********************************#   
#propagation of free particle
def Get_Analytic_Solution_free(N,a0,k0,mu,t,x):
	theta = atan(2*t/(mu*a0**2))/2.
	phase = -theta-k0**2*t/(2*mu)
	C = (2*a0**2/pi)**(1./4.)*exp(ci*phase)/(a0**4+4*t**2/mu**2)**(1./4.)

#   Cohen page 65
	Psi_analytic = C*exp(-(x-k0*t/mu)**2/(a0**2+2*ci*t/mu))

#   Normalization of Psi
	#Psi_analytic=Get_Wave_Packet_Normalization(N,Psi_analytic,delta_x)

#   standard deviation
	tmp_sigma_analytic = a0/2.*sqrt(1+4*t**2/(mu**2*a0**4))

	return Psi_analytic,tmp_sigma_analytic

#*****************************************************************************#
def Get_Analytical_Solutions_HO2D(Neig,Nx,Ny,x,y,mu,omegax,omegay):
	Hnx = np.zeros((Neig,Nx),float)
	Hny = np.zeros((Neig,Ny),float)
	Hn_coef = np.eye(Neig+1)
	#chi=np.zeros((Neig,Neig,Nx,Ny),float)

	alphax = sqrt(mu*omegax)
	alphay = sqrt(mu*omegay)
	ax = alphax*x
	ay = alphay*y

	NOx,NOy = meshgrid((np.arange(Neig)+1/2)*omegax,(np.arange(Neig)+1/2)*omegay, sparse=False, indexing='ij')
	Eanalytic = NOx+NOy

	for n in range(Neig):
		nfact = sqrt(2**n*factorial(n))
		Hnx[n,:] = hermval(ax, Hn_coef[:n+1,n])*exp(-ax**2/2)/nfact*(alphax**2/pi)**(1/4)
		Hny[n,:] = hermval(ay, Hn_coef[:n+1,n])*exp(-ay**2/2)/nfact*(alphay**2/pi)**(1/4)

	chi = einsum('nx,my->nmxy',Hnx,Hny)
	
	for nx in range(Neig):
		for ny in range(Neig):
			#   Setting bound wavefunctions to be real
			vmax = chi[nx,ny,:,:].max()
			chi[nx,ny,:,:] *= abs(vmax)/vmax

	Nx,Ny = meshgrid(range(Neig),range(Neig), sparse=False, indexing='ij')
	ind_sort = np.argsort(Eanalytic.flatten())

	return Eanalytic, chi, np.vstack((Nx.flatten()[ind_sort],Ny.flatten()[ind_sort]))

#*************************************************************#
def Get_Analytical_Solution2D(Psi,Nx,Ny,a0x,a0y,k0x,k0y,x,xc,y,yc,mu,t):
	RX,RY = np.meshgrid(x-xc,y-yc, sparse=False, indexing='ij')

	Cx = (2*a0x**2/pi)**(1./4.)
	theta_x = atan(2*t/(mu*a0x**2))/2.
	phase_x = -theta_x-k0x**2*t/(2*mu)
	tmp_ax = (a0x**4+4*t**2/mu**2)**(1./4.)

	Cy = (2*a0y**2/pi)**(1./4.)
	theta_y = atan(2*t/(mu*a0y**2))/2.
	phase_y = -theta_y-k0y**2*t/(2*mu)
	tmp_ay = (a0y**4+4*t**2/mu**2)**(1./4.)

#   Cohen page 65
	Psi_analytic = Cx*exp(ci*phase_x)/tmp_ax*exp(-(RX-k0x*t/mu)**2/(a0x**2+2*ci*t/mu))*\
				Cy*exp(ci*phase_y)/tmp_ay*exp(-(RY-k0y*t/mu)**2/(a0y**2+2*ci*t/mu))

	rms = np.einsum('ij->', abs(Psi_analytic-Psi)**2 )

	#print("time=%g rms=%g"%(t,sqrt(rms)/(Nx*Ny)))

	Delta_X_analytic = a0x/2.*sqrt(1+(2*t/(mu*a0x**2))**2)
	Delta_Y_analytic = a0y/2.*sqrt(1+(2*t/(mu*a0y**2))**2)

	return Psi_analytic,Delta_X_analytic,Delta_Y_analytic

#*******************************Perturbation transition**********************************#   
def Get_Pif_Analytical_Solutions_Approximation(W,E,P,ni,nf,delta_x,t,delta_t):
	# # 1st order
	# c1 = abs( (conjugate(P[nf,:])*P[ni,:]*delta_x) @ W @ (exp(ci*(E[nf]-E[ni])*t)*delta_t) )**2

	# 2sd order
	tmp_c2 = complex(0., 0.)
	for n in range(len(E)):
		Wfn = ((conjugate(P[nf,:]) * P[n,:] * delta_x) @ W) * exp(ci*(E[nf]-E[n])*t) * delta_t
		if n==ni: # 1st order
			c1 = abs(np.sum(Wfn))**2
		Wni = ((conjugate(P[n,:]) * P[ni,:] * delta_x) @ W) * exp(ci*(E[n]-E[ni])*t) * delta_t
		tmp_c2 += np.sum(np.triu(Wfn.reshape(len(Wfn),1) @ Wni.reshape(1,len(Wni))))
#		for itp in range(len(t)):
#			tmp_c2 += np.sum(Wni[:itp]) * Wfn[itp]

	c2 = abs(tmp_c2)**2

# #   1st order
# 	omegafi = E[nf] - E[ni]
# 	tmp_c1 = complex(0.,0.)
# 	for itp in range(len(t)):
# 		Wfi = (conjugate(P[nf,:]) * P[ni,:]) @ W[:,itp] * delta_x[0]
# 		tmp_c1 += exp(ci*omegafi*t[itp]) * Wfi * delta_t
# 	c1 = abs(tmp_c1)**2
#
# 	#   2sd order
# 	tmp_c2 = complex(0.,0.)
# 	for n in range(len(E)):
# 		omega_fn = E[nf] - E[n]
# 		omega_ni = E[n] - E[ni]
# 		for itp in range(len(t)):
# 			Wfn = (conjugate(P[nf,:]) * P[n,:]) @ W[:,itp] * delta_x[0]
# 			for itpp in range(itp):
# 				Wni = (conjugate(P[n,:]) * P[ni,:]) @ W[:,itpp] * delta_x[0]
# 				tmp_c2 += exp(ci*omega_ni*t[itpp]) * Wni * delta_t * exp(ci*omega_fn*t[itp]) * Wfn * delta_t
# 	c2 = abs(tmp_c2)**2

	return c1,c2

#*******************************Perturbation**********************************#
def Get_Analytical_Solutions_Approximation(psi0,E0,x,delta_x,W):
	N = len(x)
	Eanalytic=np.zeros((3,N),float)
	chi=np.zeros((3,N,N),float)
	W_elements=np.zeros((3,N),float)

	# Perturbation theory
	# 0th order
	Eanalytic[0,:]	= E0 #energy
	chi[0,:,:]		= psi0 #wave function
	W_elements[0,:]	= 1

#   1st order
	W_nm = (psi0*(W*delta_x)) @ psi0.T
#	W_nm = ( (psi0*(W)) @ psi0.T )*delta_x[0]
	W_nn = np.diag( W_nm )
	Eanalytic[1,:] = W_nn

	En,Em = np.meshgrid(E0,E0, sparse=False, indexing='ij'); invEnm = np.zeros(En.shape); condition = np.logical_not(En==Em); invEnm[condition] = 1/(En[condition]-Em[condition]);
	tmp = W_nm*invEnm
	chi[1,:,:]		= tmp @ psi0
	W_elements[1,:]	= np.sum(tmp,axis=1)

	# 2sd order
	Eanalytic[2,:]	= np.sum(abs(W_nm*W_nm.T)*invEnm,axis=1)
#	Eanalytic[2,:]	= np.sum(abs(W_nm)**2*invEnm,axis=1)

#	tmp = np.einsum('nl,lm,nl,nm->nm',W_nm,W_nm,invEnm,invEnm)
	tmp = ((W_nm*invEnm) @ W_nm)*invEnm
#	tmp1 = np.einsum('n,nm->nm',W_nn,W_nm*invEnm**2)
	tmp1  = np.diag(W_nn) @ (W_nm*invEnm**2)
#	tmp2 = np.einsum('nm,mn,nm->n',W_nm,W_nm,invEnm**2)/2
	tmp2 = np.sum(W_nm*W_nm.T*invEnm**2,axis=1)/2

	chi[2,:,:]		= (tmp - tmp1 - np.diag(tmp2)) @ psi0
	W_elements[2,:]	= np.sum(tmp-tmp1,axis=1)-tmp2

	return Eanalytic,chi,W_elements

#*****************************************************************************#
def Get_Analytical_Solutions_Approximation2D(Nx,Ny,Neig,psi0,E0,x,delta_x,y,delta_y,W):
	#W_elements=np.zeros((Neig,Neig),float)
	chi = np.zeros((3,Neig,Nx*Ny),float)
	Eanalytic = np.zeros((3,Neig),float)
	#
	P_W=np.zeros((Neig,Neig),float)

	dX,dY = np.meshgrid(delta_x,delta_y, sparse=False, indexing='ij')
	#dX,dY = np.meshgrid(np.append(delta_x, delta_x[0]),np.append(delta_y, delta_y[0]), sparse=False, indexing='ij')
	dS = (dX*dY).flatten()

#   Perturbation theory for degenerate initial state
	#W_elements = (psi0 @ np.diag(W.T.flatten()) @ psi0.T) *delta_x[0]*delta_y[0]	  
	W_nm = np.einsum( 'mx,x,nx->mn', psi0, W.flatten()*dS, psi0 ) # <m|W|n>
	W_elements = W_nm

	#Eanalytic[0,:], P_W = LA.eigh(W_elements) #energy of the 1st order



#   0th order
	Eanalytic[0,:]	= E0 #energy
	chi[0,:,:]		= psi0 #wave function

#   1st order
	W_nn = np.diag( W_nm )
	Eanalytic[1,:] = W_nn #energy of the 1st order
	En,Em = np.meshgrid(E0,E0, sparse=False, indexing='ij'); invEnm = np.zeros(En.shape); condition = np.logical_not(En==Em); invEnm[condition] = 1/(En[condition]-Em[condition]);
	tmp = W_nm*invEnm
	chi[1,:,:]		= tmp @ psi0

#   2sd order
	Eanalytic[2,:]	= np.sum(abs(W_nm*W_nm.T)*invEnm,axis=1)
	tmp = ((W_nm*invEnm) @ W_nm)*invEnm
	tmp1  = np.diag(W_nn) @ (W_nm*invEnm**2)
	tmp2 = np.sum(W_nm*W_nm.T*invEnm**2,axis=1)/2
	chi[2,:,:]		= (tmp - tmp1 - np.diag(tmp2)) @ psi0

	norm = abs(np.sum((chi[0,:,:]+chi[1,:,:]+chi[2,:,:])**2*dS,axis=1))




	#chi[0,:,:] = P_W @ psi0 # wave function of the 1st order
	##for m in range(Neig):
		##chi[:,m,0]=0.
		##for n in range(Neig):
			##chi[:,m,0] += P_W[n,m]*psi0[n,:] # wave function of the 1st order

	#norm = abs(np.sum(chi[1,-1,:]**2*dS))

	#norm=0.
	#for i in range(Nx):
		#for j in range(Ny):
			#chi_plot[i,j]=chi[i+Nx*j,Neig-1,0]
			#norm += abs(chi_plot[i,j])**2*delta_x*delta_y

	print('Checkout the normalization of chi_{n=%d}(x,y)='%(Neig-1),norm)

	return Eanalytic,chi,P_W
