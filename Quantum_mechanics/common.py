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

from time import process_time
import numpy as np

#*****************************************************************************#
def print_E(E,E_analytic,Neig):
	print ('<br>       n         E          Eanalytique      |E-Eanalytique|*100/Eanalytique ')
	for n in range(min([Neig,len(E),len(E_analytic)])):
		error=abs(E[n]/E_analytic[n]-1)*100
		if(E[n]<0.):
			print ('<br>  %8d  %14f  %14f %20g ' % ( n, E[n], E_analytic[n], error) )


#*************************************************************#
def func_timetest(fname1,arg1,fname2=None,arg2=None):

	wtime1 = process_time( )
	out1 = fname1(*arg1)
	wtime1 = process_time( )-wtime1
	print('<br>\tt1=%g s for %s'% (wtime1,fname1.__name__))
	if fname2==None:
		return
	wtime2 = process_time( )
	out2 = fname2(*arg2)
	wtime2 = process_time( )-wtime2
	print('<br>\tt2=%g s (ratio=%g) for %s'% (wtime2,wtime2/wtime1,fname2.__name__))


	if type(out1) is tuple:
		if not len(out1)==len(out2):
			print('<br>Output of %s and %s functions is fifferent.'%(fname1,fname2))
			return
		for i in range(len(out1)):
			print('<br>\t\tnorm of output ',i,' is\t',np.linalg.norm( out1[i]-out2[i] ))
	else:
		print('<br>\t\tnorm of single output is\t',np.linalg.norm( out1-out2 ))
#		print('<br>\t\tout1=',out1)
#		print('<br>\t\tout2=',out2)

