---
title:
- High-Order Accuracy Coumputation of Coupling Functions for Strongly Coupled Oscllators (Documentation)
...

---
author:
- Youngmin Park
...

# Introduction

StrongCoupling is a script for computing the higher-order coupling functions in my paper with Dan Wilson, ``High-Order Accuracy Compuation of Coupling Functions for Strongly Coupled Oscillators''

## Dependencies

All following libraries are required to make the script run.

| Package	| Version	| Link		| 
| -----------	| -----------	| -----------	|
| Python	| 3.7.7		|
| Matplotlib	| 3.3.1		|		|
| Numpy		| 1.19.1	|		|
| Scipy		| 1.5.2		|		|
| Pathos	| 0.2.6		| https://anaconda.org/conda-forge/pathos |
| tqdm		| 4.48.2	| https://anaconda.org/conda-forge/tqdm |
| Sympy		| 1.6.2		| https://anaconda.org/anaconda/sympy |

Notes on depedendencies:

**Python 3.7+ is necessary**. Our code often requires more than 256 function inputs. Python 3.6 or earlier versions have a limit of 256 inputs and will not work with our scripts. The script will likely work with earlier versions of all other libraries.

### Other Notes

I intentially chose **pathos** over multiprocessing because pickling is more robust with pathos. Pathos uses dill, which can serialize far more objects compared to multiprocessing, which uses pickle.

The code is written so that tqdm is necessary, but tqdm only provides a status bar during parallel computing. It is not part of the engine, and the code can be modified to work without it. In future versions I may leave tqdm as a toggle.

## Installation

As long as your computer has the packages listed above and they are installed using Python 3.7, the StrongCoupling script should run.

I will not release the StrongCoupling script as an installable package simply because I do not have to time to maintain and track version releases for distribution platforms such as anaconda, pip, and apt. Worst case scenario, ``yarrr matey'' a Windows 10 virutal machine and install everything using anaconda.

Note that in Ubuntu a virtual environment should be used to run the StrongCoupling script. Ubuntu uses Python 3.6 by default and does not like it when the default is changed to Python 3.7.

# Set up a Model

Let's walk through setting up a script using the Goodwin oscillator ([Gonze et al 2005](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1366510/)):

$$\frac{dX_i}{dt} = \nu_1 \frac{K_1^n}{K_1^n+Z_i^n} - \nu_2 \frac{X_i}{K_2 + X_i}+ \nu_c \frac{KF}{K_c + KF} + L,$$
$$\frac{dY_i}{dt} = k_3 X_i - \nu_4 \frac{Y_i}{K_4 + Y_i},$$
$$\frac{dZ_i}{dt} = k_5 Y_i - \nu_6 \frac{Z_i}{K_6 + Z_i}.$$

The neurotransmitter concentration satisfies:

$$\frac{dV_i}{dt} = k_7 X_i - \nu_8 \frac{V_i}{K_8 + V_i},$$

and the coupling $F$ is defined as

$$F = \frac{1}{N}\sum_{i=1}^N V_i.$$

To use StrongCoupling, define a right-hand side function with the name ``rhs'', and a coupling function named ``coupling''. rhs is defined for a single oscillator:

```python
def rhs(t,z,pdict,option='value'):
	x,y,z,v = z
	
	n = pdict['n']
	
	nu1 = pdict['nu1']
	nu2 = pdict['nu2']
	nu4 = pdict['nu4']
	nu6 = pdict['nu6']
	nu8 = pdict['nu8']
	
	k1n = pdict['k1']**n
	k2 = pdict['k2']
	k3 = pdict['k3']
	k4 = pdict['k4']
	k5 = pdict['k5']
	k6 = pdict['k6']
	k7 = pdict['k7']
	k8 = pdict['k8']
	
	L = pdict['L']
	
	dx = nu1*k1n/(k1n+z**n) - nu2*x/(k2+x) + L
	dy = k3*x - nu4*y/(k4+y)
	dz = k5*y - nu6*z/(k6+z)
	dv = k7*x - nu8*v/(k8+v)
	
	if option == 'value':
		return np.array([dx,dy,dz,dv])
	elif option == 'sym':
		return Matrix([dx,dy,dz,dv])
```

The input t is time and the input z is an array or list containing all the variables. pdict contains all the parameters in a dictionary. It is important to pdict treat as a dict because the rhs function plays two roles. One, the rhs function is put into scipy ODE solvers, in which case pdict contains key-value pairs of parameters and floats. Two, we sometimes take symbolic derivatives of the rhs function, in which case pdict contains key-value pairs of parameters and sympy objects. The option input is used by StrongCoupling to help the rhs function return the correct format for numeric or symbolic manipulation.

The coupling fnction is defined for a single oscillator from the perspective of the first oscillator:

```python
def coupling(vars_pair,pdict,option='value'):
	x1,y1,z1,v1,x2,y2,z2,v2 = vars_pair
	
	K = pdict['K']
	vc = pdict['eps']
	F = 0.5*(v1+v2)
	
	if option == 'value':
		return np.array([vc*K*F,0,0,0])
	elif option == 'sym':
		return Matrix([vc*K*F,0,0,0])
```

Note that the parameter vc is the coupling parameter in the paper, so we let it take the value of the parameter epsilon. Next, define a main() function where we define the variables, parameter dictionary, keyword options, limit cycle initial condition, and the StrongCoupling call.

```python
def main():
	var_names = ['x','y','z','v']
    pardict = {'v1_val':.84,'v2_val':.42,'v4_val':.35,'v6_val':.35,'v8_val':1,
               'k1_val':1,'k2_val':1,'k3_val':.7,'k4_val':1,'k5_val':.7,
               'k6_val':1,'k7_val':.35,'k8_val':1,'K_val':0.5,'kc_val':1,
               'n_val':6,'L_val':0,'eps_val':0}
    
    kwargs = {'g_forward':True,'z_forward':False,'i_forward':False,
              'i_bad_dx':[False,True,False,False,False,False],
              'dense':True,
              'dir':'home+goodwin_dat/',
              'trunc_order':5,
              'trunc_deriv':5,
              'NA':2000,
              'p_iter':20,
              'max_iter':200,
              'TN':2000,
              'rtol':1e-13,
              'atol':1e-13,
              'rel_tol':1e-10,
              'method':'LSODA'}
			  
	T_init = 23.54
    LC_init = np.array([.1734,.39,1.8814,.2708,T_init])
    
    a = StrongCoupling(rhs,coupling,LC_init,var_names,pardict,**kwargs)
```

Please see the StrongCoupling class below for more details on these keyword arguments. It is not necessary to define the kwargs dict as shown above -- keyword arguments may be entered in the standard keyword=value format straightt into the StrongCoupling class. I prefer to keep the kwargs options explicit in a dictionary because I may change the defaults.

The parameters
    g_forward, z_forward, i_forward
test

The limit cycle initial condition must be estimated by hand. For example, I used XPP and simulated the limit cycle for a long time, then extracted x, y, z, v values at some large time value. The XPP file for this oscillator is contained in ode/goodwin.ode. The initial condition and period do not need to be precise because the code will search for the limit cycle via Newton's method, but closer is better.



