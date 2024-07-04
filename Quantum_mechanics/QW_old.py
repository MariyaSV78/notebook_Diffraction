from __future__ import division
import numpy as np

# old - non opptimized




def Get_Kinetic_Opertor_old(N,dx,m):
	LBox=dx*(N)
	coef=pi*pi/(m*LBox*LBox)
	if(N%2==0):
		coefdiag=coef*(N*N+2.)/6.
		
		for i in range(N):
			T[i,i]=coefdiag
			
			for j in range(i+1,N):
				s=1.0/(sin((j-i)*pi/N))**2
				T[i,j]=(-1)**(i-j)*coef*s
				T[j,i]=T[i,j]
	else:
		coefdiag=coef*(N*N-1)/6.
		for i in range(N):
			T[i,i]=coefdiag
			
			for j in range(i+1,N):
				s=cos((j-i)*pi/N)/(sin((j-i)*pi/N))**2
				T[i,j]=(-1)**(i-j)*coef*s
				T[j,i]=T[i,j]

#*****************************************************************************#   
def Get_Analytical_Solutions_delta_old(N,x,x0,xL,m,V0,lambda_0,N_analytic):
	epsilon=1e-5
	ns=0
	e=np.zeros(N_analytic,float)
	e=linspace(V0,0.,N_analytic)
	E_analytic = np.zeros(N,float)

	for i in range(N_analytic-1):
		a=e[i]
		b=e[i+1]

		E_analytic[ns] = Get_Zero_Dichotomique(a,b,lambda_0,epsilon,0)
		E_analytic[ns] = Get_Zero_Dichotomique(a,b,lambda_0,epsilon,1)
		ns=ns+1

#	for i in range(N_analytic-1):
#		a=e[i]
#		b=e[i+1]

#		E_analytic[ns]=Get_Zero_Dichotomique(a,b,epsilon,1)
#		ns=ns+1



	for i in range(ns):
		for j in range(i+1,ns):
			if(E_analytic[i]>E_analytic[j]):
				tmpc=E_analytic[i]
				E_analytic[i] = E_analytic[j]
				E_analytic[j] = tmpc

	return E_analytic,ns

	figure(5)
	plot(e,sqrt(2*m*(e-V0))*tan(a*sqrt(2*m*(e-V0))),'-b',e,sqrt(-2*m*e),'-r')

	figure(6)
	plot(e,sqrt(2*m*(e-V0))*1./tan(a*sqrt(2*m*(e-V0))),'-b',e,-sqrt(-2*m*e),'-r')


	figure(5)
	plot(q,q*lambda_0-1,'--k',q,exp(-q*d),'-r',q,-exp(-q*d),'-b')
	legend(["$q\lambda_0-1$"],loc=9)
	plt.plot(tmp_q0,tmp_q0*lambda_0-1,'ok')
	plt.plot(tmp_q1,tmp_q1*lambda_0-1,'ok')
	annotate('$e^{-qd}$, pair', xy=(0.4,0.3),xytext=(0.4,0.3),xycoords='data',size=18)
	annotate('$-e^{-qd}$, impair',xy=(1,-0.2),xytext=(1,-0.2),xycoords='data',size=18)
	xlim(0,2)
	ylim(-1,1)
	xlabel("$q$",size=20)
	ylabel("Solutions analytiques pairs et impairs",size=15)
	savefig('solutions_analytiques_double_potentiel_delta.eps')

#*****************************************************************************#
def Get_WKB_old(E,N,r,r1,r2,m,V,delta_r,tauF,v_plot):

	phase=0.
	for i in range(N):
		if(r[i]>r1 and r[i]<r2):
			phase=phase+sqrt(2*m*(real(V[i])-real(E[v_plot])))*delta432_r[i]

	print('<br>phase=%g   T_WK=%g '%(phase,exp(-phase)**2))
	print('<br>Gamma_WKB=%g  1/s'%(exp(-phase)**2/tauF))
	print('<br>halftime=%g  s'%(tauF/exp(-phase)**2))
	print('<br>ln(2)/Gamma=%g  s'%(log(2)/(exp(-phase)**2/tauF)))

	return phase

#*****************************************************************************#

def Get_Current_old(psi,N,m,delta_x):
	grad_psi=0.
	grad_psi_star=complex(0.,0.)

	for j in range(N):
		for i in range(N-1):
			grad_psi=(psi[i+1,j]-psi[i,j])/(delta_x)
			grad_psi_star=(conjugate(psi[i+1,j])-conjugate(psi[i,j]))/(delta_x)
			J[i,j]=ic/(2*m)*(psi[i,j]*grad_psi_star-conjugate(psi[i,j])*grad_psi)

		J[N-1,j]=J[N-2,j]

	return J


def Get_Heisenberg_Uncertainty_old(x,delta_x,psi):
	N = len(x)
	grad_psi = 0.
	x_average = np.zeros(N,float)
	p_average = np.zeros(N,float)
	Delta_X = np.zeros(N,float)
	Delta_P = np.zeros(N,float)
	Heisenberg = np.zeros(N,float)

	for n in range(N):
		x_average[n]=0.
		xsquare_average=0.
		p_average[n]=0.
		tmp_p_average=complex(0.,0.)
		psquare_average=0.
	
		for i in range(N):
			x_average[n] += psi[i,n]*psi[i,n]*x[i]*delta_x[0]
			xsquare_average= xsquare_average+psi[i,n]*psi[i,n]*x[i]**2*delta_x[0]

		Delta_X[n]=sqrt(abs(xsquare_average-x_average[n]**2))
		
		for i in range(N-1):
			grad_psi=(psi[i+1,n]-psi[i,n])/(delta_x)
			tmp_p_average -= ci*grad_psi*psi[i,n]*delta_x[0]
		p_average[n]=abs(tmp_p_average)

		for i in range(1,N-1):
			laplacien_psi=(psi[i+1,n]-2*psi[i,n]+psi[i-1,n])/(delta_x[0]**2)
			psquare_average=psquare_average-laplacien_psi*psi[i,n]*delta_x[0]
	
		Delta_P[n]=sqrt(abs(psquare_average-p_average[n]**2))
		Heisenberg[n]=Delta_X[n]*Delta_P[n]

	return Heisenberg,x_average,p_average,Delta_X,Delta_P



#*****************************************************************************#
def Get_Barrier_Transimission_old(N,E,V,m,n,r,r1,r2,i1,i2,v_plot,psi,tauF,V_Model,maxloc):

	i2=N-1
	for i in range(N-1,0,-1):
		if(r[i]>r[maxloc] and real(E[v_plot])>V_Model[i]):
			i2=i
	i1=0
	for i in range(N):
		if(r[i]<=r[maxloc] and real(E[v_plot])>V_Model[i]):
			i1=i
	print("<br>i1=",i1)
	print("<br>i2=",i2)
	print("<br>maxloc=",maxloc)
	print("<br>v_plot=",v_plot)

	ii2 = logical_and(r[1:-1]>r[maxloc], real(E[v_plot])>V_Model[1:-1]).nonzero().max()
	ii1 = logical_and(r>r[maxloc], real(E[v_plot])>V_Model).nonzero().max()
	print("<br>ii1=",i1)
	print("<br>ii2=",i2)


	print("<br>4 i1=",i1)
	print("<br>4 i2=",i2)
	print("<br>4 v_plot=",v_plot)

	In = int((i2-i1)/n)
	Delta=r[In]-r[0]
	Vi=np.zeros(n+1,float)
	ri=np.zeros(n+1,float)
	Vi_plot=np.zeros((N,n+1),float)
	ri_plot=np.zeros((N,n+1),float)
	Transmission=np.zeros(n,float)
	psi_plot=np.zeros(5000,complex)
	r_plot=linspace(r.min(),r.max(),5000)
	psi_plot=interp1d(r, psi[:,v_plot], kind='cubic')

	print("<br>Delta=%g fm"%(Delta*au_to_fm)) 
	index=0
	print("<br>3 i1=",i1)
	print("<br>3 i2=",i2)
	print("<br>3 In=",In)
	for i in range(i1+1,i2,In):
		print("<br>index=",index)
		Vi[index]=(real(V_Model[i])+real(V_Model[i+In]))/2.
		ri[index]=r[i]
		index=index+1

	print('<br>  Barrier       Height       Transmission')
	Total_Transmission=1.
	for i in range(n):
		Transmission[i]=exp(-2*sqrt(2*m*(Vi[i]-real(E[v_plot])))*Delta)
		Total_Transmission=Total_Transmission*Transmission[i]
		print('<br>    %d      %g         %g  '%(i,Vi[i]*au_to_MeV,Transmission[i]))

	print('<br>Total Transmission for %d segments %g is '%(n,Total_Transmission))

	for j in range(n):
		for i in range(N):
			if(r[i]>ri[j] and r[i]<ri[j+1]):
				Vi_plot[i,j]=Vi[j]
			ri_plot[i,j]=r[i]
		  
	return Total_Transmission,ri_plot,Vi_plot

	#    figure(-2)
	plot(r_plot*au_to_fm,real(E[v_plot])*au_to_MeV+real(psi_plot(r_plot))*10,'-k',lw=2)
	for j in range(n):
		plot(ri_plot[:,j]*au_to_fm,Vi_plot[:,j]*au_to_MeV,lw=2)
	   
	annotate('$r_1$', xy=(r[i1]*au_to_fm+1.,8),xytext=(r[i1]*au_to_fm+1.,8),xycoords='data',size=20,color='k') 
	annotate('$r_2$', xy=(r[i2]*au_to_fm,real(E[v_plot])*au_to_MeV*1.1),xytext=(r[i2]*au_to_fm,real(E[v_plot])*au_to_MeV*1.1),xycoords='data',size=20,color='k') 

	print('<br>half-time=%g'%(tauF/Total_Transmission))
	#    savefig('Gamow.eps')
	savefig('../chap3_5.png')
	
def Get_Fourrier_Transform_WF_old(k,r,r2,delta_r,psi):
	if len(psi.shape)==1:
		psi = psi.reshape((len(psi), 1))

	N=len(r)
	Nk=len(k)
	psi_FT=np.zeros((Nk,psi.shape[1]),complex)

	for j in range(Nk):
		for i in range(N):
			if(r[i]>r2):
				psi_FT[j,:] += exp(-ci*r[i]*k[j])*psi[i,:]*delta_r[i]*au_to_fm/sqrt(2*pi)

	return psi_FT

def Get_Analytical_Solutions_square_old(N,x,x0,xL,m,V0,Nanalytic):
	epsilon=1e-3
	ns=0
	z0=L*sqrt(2*m*V0)
	print('z0=',z0)
	z=linspace(0.,z0,Nanalytic)
	h=np.zeros(Nanalytic,float)
	h=tan(z)

	for i in range(Nanalytic):
		if(abs(h[i])>100.):
			h[i]=100.

	for i in range(Nanalytic-1):
		z1=z[i]
		z2=z[i+1]
		e1=z1**2/(L**2*2*m)-V0
		e2=z2**2/(L**2*2*m)-V0
		if((i%2==0 or i==0) and e1 <0. and e2<0. and f1(z1)*f1(z2)<0. and E[ns]<0):
			#tmp_z=op.newton(f1,z1,df1, args=(), tol=1.48e-03, maxiter=100)
			#tmp_z=op.bisect(f1,z1,z2)
			tmp_z=op.fsolve(f1,z1)
			tmp_E=tmp_z**2/(L**2*2*m)-V0

			if(abs(f1(tmp_z))<epsilon):
				Eanalytic[ns]=tmp_E
				z_analytic[ns]=tmp_z
				print('paire',i,tmp_z,f1(tmp_z))
				ns=ns+1
		elif(((i+1)%2==0) and e1 <0. and e2<0. and f2(z1)*f2(z2)<0. and E[ns]<0):
			#tmp_z=op.newton(f2,z1,df2, args=(), tol=1.48e-03, maxiter=100)
			#tmp_z=op.bisect(f2,z1,z2)
			tmp_z=op.fsolve(f2,z1)
			tmp_E=tmp_z**2/(L**2*2*m)-V0
			
			
			if(abs(f2(tmp_z))<epsilon):
				Eanalytic[ns]=tmp_E
				z_analytic[ns]=tmp_z
				print('imp',i,tmp_z,f2(tmp_z))
				ns=ns+1

	return Eanalytic,z_analytic,ns

	figure(5)
	plot(z,h,'-k',z,sqrt((z0/z)**2-1),'--r',lw=1)
	plt.plot(z_analytic[0],sqrt((z0/z_analytic[0])**2-1),'ok',lw=3) 
	plt.plot(z_analytic[2],sqrt((z0/z_analytic[2])**2-1),'ok',lw=3) 
	plt.plot(6.8,sqrt((z0/6.8)**2-1),'ok') 
	legend(["$tan(z)$", "$\sqrt{(z_0/z)^2-1}$"])
	ylabel("Solution graphique des etats pairs ",size=20)
	xlabel("$z$",size=20)
	ylim(-1,10)
	savefig('potentiel_carre_analytique_paire.eps')
	
	figure(6)
	plot(z,-1./h,'-k',z,sqrt((z0/z)**2-1),'--r',lw=1)
	plt.plot(z_analytic[1],sqrt((z0/z_analytic[1])**2-1),'ok',lw=3) 
	plt.plot(z_analytic[3],sqrt((z0/z_analytic[3])**2-1),'ok',lw=3) 
	legend(["$-cot(z)$", "$\sqrt{(z_0/z)^2-1}$"],loc=9)
	ylabel("Solution graphique des etats impairs",size=20)
	xlabel("$z$",size=20)
	ylim(-1,10)
	savefig('potentiel_carre_analytique_impaire.eps')

#*******************************Perturbation**********************************#   
def Get_Analytical_Solutions_Approximation_old(N,Neig,psi0,E0,x,delta_x,W):
#   Perturbation theory
#	for n in range(Neig):
#   0th order
	Eanalytic=np.zeros((3,N),float)
	chi=np.zeros((3,N,N),float)
	W_elements=np.zeros((3,N),float)

#   1st order
	Eanalytic[0,:]=E0 #energy
	chi[0,:,:] = psi0 #wave function
	W_elements[0,:]=1

	for n in range(N):
		W_elements_0=0.
		for i in range(N): # energy
			W_elements_0 += psi0[n,i]*W[i]*psi0[n,i]*delta_x[0]
		Eanalytic[1,n] = W_elements_0

		for m in range(N):# wave function
			if(m!=n):
				W_elements_0=0.
				for i in range(N):
					W_elements_0 += psi0[m,i]*W[i]*psi0[n,i]*delta_x[0]
				chi[1,n,:] += W_elements_0/(E0[n]-E0[m])*psi0[m,:]
				W_elements[1,n] += W_elements_0/(E0[n]-E0[m])

#   2sd order
		for m in range(N):# energy
			if(m!=n):
				W_elements_1=0.
				for i in range(N):
					W_elements_1 += psi0[m,i]*W[i]*psi0[n,i]*delta_x[0]
				Eanalytic[2,n] += abs(W_elements_1)**2/(E0[n]-E0[m])

		for m in range(N):# wave function correction 1
			for l in range(N):
				if(m!=n and l!=n):
					W_elements_1=0.;W_elements_2=0.
					for i in range(N):
						W_elements_1 += psi0[m,i]*W[i]*psi0[l,i]*delta_x[0]
						W_elements_2 += psi0[l,i]*W[i]*psi0[n,i]*delta_x[0]
					chi[2,n,:] += W_elements_1*W_elements_2/(E0[n]-E0[m])/(E0[n]-E0[l])*psi0[m,:]
					W_elements[2,n] += W_elements_1*W_elements_2/(E0[n]-E0[m])/(E0[n]-E0[l])

		for m in range(N): #wave function corrections 2 and 3
			if(m!=n):
				W_elements_1=0.;W_elements_2=0. # wave function correction 2
				for i in range(N):
					W_elements_1 += psi0[n,i]*W[i]*psi0[n,i]*delta_x[0]
					W_elements_2 += psi0[m,i]*W[i]*psi0[n,i]*delta_x[0]
				chi0[2,n,:] -= W_elements_1*W_elements_2/(E0[n]-E0[m])**2*psi0[m,:]
				W_elements[2,n] -= W_elements_1*W_elements_2/(E0[n]-E0[m])**2

				W_elements_1=0.;W_elements_2=0.# wave function correction 3
				for i in range(N):
					W_elements_1 += psi0[n,i]*W[i]*psi0[m,i]*delta_x[0]
					W_elements_2 += psi0[m,i]*W[i]*psi0[n,i]*delta_x[0]
				chi[2,n,:] -= (1./2.)*W_elements_1*W_elements_2/(E0[m]-E0[n])**2*psi0[n,:]
				W_elements[2,n] -= (1./2.)*W_elements_1*W_elements_2/(E0[m]-E0[n])**2


	return Eanalytic,chi,W_elements

