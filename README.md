
---
title: "Generating Higher-Order Coupling Functions for Strongly Coupled Oscillators: A Python Library"
author: "Youngmin Park and Dan Wilson"
output: pdf_document
abstract: >
  We introduce several detailed examples of how to use the StrongCoupling library. The framework is reasonably general, with no a priori restrictions on model dimension or type of coupling function. We only require differentiability. Examples in this document include the Goodwin oscillator of circadian rhythms, and a small coupled system of two chemical oscillators. 

...



---
author:

...



# Introduction

StrongCoupling is a script for computing the higher-order coupling functions in my paper with Dan Wilson, ``High-Order Accuracy Compuation of Coupling Functions for Strongly Coupled Oscillators''. The script generates higher-order interaction functions for phase reductions of systems containing limit cycle oscillations.

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

I chose **pathos** over multiprocessing because pickling is more robust with pathos. Pathos uses dill, which can serialize far more objects compared to multiprocessing, which uses pickle.

The code is written so that tqdm is necessary, but tqdm only provides a status bar during parallel computing. It is not part of the engine, and the code can be modified to work without it. In future versions I may leave tqdm as a toggle.

## Installation

As long as your computer has the packages listed above and they are installed using Python 3.7, the StrongCoupling script should run. Just place it within the same working directory as your Python script and import it as a module.

I have no immediate plans to release the StrongCoupling script as an installable package simply because I do not have the time to maintain and track version releases for distribution platforms such as anaconda, pip, and apt.

Note that in Ubuntu a virtual environment should be used to run the StrongCoupling script. Ubuntu uses Python 3.6 by default and does not like it when the default is changed to Python 3.7.

# Reproduce Figures

To reproduce the figures in Park and Wilson 2020, cd to the examples directory and run

   $ generate_figures.py

This file will call the complex Ginzburg-Landau (CGL) model file (CGL.py) and the thalamic model file (Thalamic.py) and generate the figures from the paper. It will take a while to run, and will use 4 cores by default! Make sure to edit the keyword arguments (documented in the StrongCoupling section below) if you wish to use more or less cores.


# Set up a Model: Goodwin Oscillator

Let's walk through setting up a script using the Goodwin oscillator ([Gonze et al 2005](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1366510/)):

$$\frac{dX_i}{dt} = \nu_1 \frac{K_1^n}{K_1^n+Z_i^n} - \nu_2 \frac{X_i}{K_2 + X_i}+ \nu_c \frac{KF}{K_c + KF} + L,$$
$$\frac{dY_i}{dt} = k_3 X_i - \nu_4 \frac{Y_i}{K_4 + Y_i},$$
$$\frac{dZ_i}{dt} = k_5 Y_i - \nu_6 \frac{Z_i}{K_6 + Z_i}.$$

The neurotransmitter concentration satisfies:

$$\frac{dV_i}{dt} = k_7 X_i - \nu_8 \frac{V_i}{K_8 + V_i},$$

and the coupling $F$ is defined as

$$F = \frac{1}{N}\sum_{i=1}^N V_i.$$

## Right-hand Side and Coupling Function

To use StrongCoupling, define a right-hand side function with the name ``rhs'', and a coupling function named ``coupling''. rhs is defined for a single oscillator:

```python
def rhs(t,z,pdict,option='value'):
	x,y,z,v = z
    
    p = pdict
    n = p['n']
    
    dx = p['v1']*p['k1']**n/(p['k1']**n+z**n) - p['v2']*x/(p['k2']+x) + p['L']
    dy = p['k3']*x - p['v4']*y/(p['k4']+y)
    dz = p['k5']*y - p['v6']*z/(p['k6']+z)
    dv = p['k7']*x - p['v8']*v/(p['k8']+v)
    
    if option == 'value':
        return np.array([dx,dy,dz,dv])
    elif option == 'sym':
        return Matrix([dx,dy,dz,dv])
```

The input t is time and the input z is an array or list containing all the variables. pdict contains all the parameters in a dictionary. It is important to pdict treat as a dict because the rhs function plays two roles. One, the rhs function is put into scipy ODE solvers, in which case pdict contains key-value pairs of parameters and floats. Two, we sometimes take symbolic derivatives of the rhs function, in which case pdict contains key-value pairs of parameters and sympy objects. The option input is used by StrongCoupling to help the rhs function return the correct format for numeric or symbolic manipulation.

The coupling function is also defined for a single oscillator from the perspective of the first oscillator:

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

## Set Up Keyword Arguments

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
              'method':'LSODA',
              'processes':4,
              'chunksize':10000}
			  
	T_init = 23.54
    LC_init = np.array([.1734,.39,1.8814,.2708,T_init])
    
    a = StrongCoupling(rhs,coupling,LC_init,var_names,pardict,**kwargs)
```

Please see the StrongCoupling class below for more details on these keyword arguments. It is not necessary to define the kwargs dict as shown above -- keyword arguments may be entered in the standard keyword=value format straight into the StrongCoupling class.

Assuming that the appropriate libraries have been installed, you should be able to run this script now:

    $ python3 goodwin.py

When the program runs, it will attempt to find the limit cycle. The limit cycle initial condition must be estimated by hand. For example, I used XPP and simulated the limit cycle for a long time, then extracted x, y, z, v values at some large time value. The XPP file for this oscillator is contained in ode/goodwin.ode. The initial condition and period do not need to be precise because the code will search for the limit cycle via Newton's method, but closer is better.

### Computing Floquet Eigenfunctions, PRCs, and IRCs

It will then solve for the one Floquet multiplier (assuming there is only one slowest decaying mode) and turn to computing the hierarchy of ODEs $g^{(k)}$, $z^{(k)}$, and $i^{(k)}$ in that order. To solve for the hierarchy, the parameters

    g_forward, z_forward, i_forward
	
tell the ODE solver to solve in forwards or backward time. With the options above, the integrator will solve for $g^{(k)}$ forwards in time for all $k$ and solve for $z^{(k)}$ and $i^{(k)}$ backwards in time for all $k$. These choices must be determined by hand, by checking whether or not the Newton's method converges. It is an involved process.

The parameter

    i_bad_dx

makes Newton's method include an additional variable when convergence is weak. This parameter depends on the system. This is also an involved process and can only be decided after attempting to solve the ODEs. I will add more on this another time.

The parameters 

    rtol, atol, TN, method, dense, rel_tol
	
are additional numerical parameters, most of which go into solve_ivp. rtol and atol determine the absolute and relative tolerance. I like to keep these values small. TN is the total number of time steps to be used in the 1d interpolation calls. dense is the keyword argument that goes into solve_ivp and determines whether the solver will use a predefined time mesh or a time mesh determined by solve_ivp. rel_tol is the convergence threshold for Newton's method. 

It will take some time for the script to solve the symbolic equations for $g^{(k)}$, $z^{(k)}$, and $i^{(k)}$ then it will take a longer time for the method to evaluate them numerically. rtol and atol determine the speed of the numerical integration, and they may be increased to speed up the integration depending on the system and the desired order of $\mathcal{H}$. Problems that do not need fine time steps to solve for $g^{(k)}$, $z^{(k)}$, and $i^{(k)}$ can afford to use larger rtol and atol.

As the script generates $g^{(k)}$, $z^{(k)}$, and $i^{(k)}$ functions, it will save plots to the current working directory (plot\*.png). These plots are for debugging purposes, so that the user can check whether or not functions are converging and decide if the numerical choice above are appropriate. If you wish to save these plots, please rename or copy/paste them to another directory. The StrongCoupling will overwite the plots when any example is run. 

The symbolic files and data files for $g^{(k)}$, $z^{(k)}$, and $i^{(k)}$ are saved to

    home+goodwin_dat/
	
File names 	

	*_data*_TN_*.txt 
	
Contain the trajectories for $g^{(k)}$, $z^{(k)}$, and $i^{(k)}$. Symbolic functions are saved to 

    *.d
	
files.

### Computing Generalized Interaction Functions

The parameters

    processes, chunksize

set the number of processors to use for multiprocessing and the number of 'jobs' to give to each processor when computing $p^{(k)}$. There is no hard and fast rule for these values. Please see the other example files to get an idea for how chunksize can be set.

The script will then generate the symbolic coupling functions and evaluate them to generate the $p^{(k)}$ functions using the 4 cores specified earlier, then generate the $\mathcal{H}^{(k)}$ functions up to 5th order. The data for the $\mathcal{H}^{(k)}$ functions will be saved to 

    home+goodwin_dat/

i.e., the goodwin_dat folder in the home directory, with the file names

    h_dat_*_NA=2000_piter=20.txt

## Plotting Data

It is up to you to decide how to use these files. To plot the files, feel free to append the following lines to the main() function:

```python
	# plot H functions
    phi = np.linspace(0,a.T,a.NA)
    for k in range(a.trunc_order+1):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(phi,a.hodd['dat'][k])
        ax.set_title('hodd'+str(k)+' NA='+str(a.NA))
        plt.show(block=True)
```


# Set up a Model: Chemical Star Network (UNDER CONSTRUCTION)

Let's try setting up the model from [Norton et al 2019](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.148301). We will focus on the case of two oscillators for now. From the supplementary information, the equations are:

$$\frac{x}{dt} = -k_1xy + k_2 y - 2k_3 x^2 + k_4\frac{x(c_0-z)}{(c_0-z+c_{min})},$$
$$\frac{y}{dt} = -3k_1xy - 2k_2y - k_3x^2 + k_7 u + k_9 z + k_I \frac{(c_0-z)}{b_c/b+1},$$
$$\frac{z}{dt} = 2k_4 \frac{x(c_0-z)}{c_0 -z+c_{min}} - k_9 z - k_{10} z + k_I \frac{c_0 - z}{b_c/b+1},$$
$$\frac{u}{dt} = 2k_1 xy + k_2y + k_3 x^2 - k_7 u,$$

and the coupling $F$ is defined as

$$F = \frac{1}{N}\sum_{i=1}^N V_i.$$



## Right-hand Side and Coupling Function

As above, we define a right-hand side function with the name ``rhs'', and a coupling function named ``coupling''. rhs is defined for a single oscillator:

```python
def rhs(t,z,pdict,option='value'):
	x,y,z,v = z
    
    p = pdict
    n = p['n']
    
    dx = p['v1']*p['k1']**n/(p['k1']**n+z**n) - p['v2']*x/(p['k2']+x) + p['L']
    dy = p['k3']*x - p['v4']*y/(p['k4']+y)
    dz = p['k5']*y - p['v6']*z/(p['k6']+z)
    dv = p['k7']*x - p['v8']*v/(p['k8']+v)
    
    if option == 'value':
        return np.array([dx,dy,dz,dv])
    elif option == 'sym':
        return Matrix([dx,dy,dz,dv])
```



