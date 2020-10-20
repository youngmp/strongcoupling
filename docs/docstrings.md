---
description: |
    API documentation for modules: StrongCoupling, Thalamic, CGL.

lang: en

classoption: oneside
geometry: margin=1in
papersize: a4

linkcolor: blue
links-as-notes: true
...


    
# Module `StrongCoupling` {#StrongCoupling}

StrongCoupling.py computes the higher-order interaction functions from
Park and Wilson 2020 for $N=2$ models and one Floquet multiplier.
plt.
In broad strokes, this library computes functions in the following order:

* Use the equation for $\Delta x$ (15) to produce a hierarchy of
ODEs for $g^{(k)}$ and solve. (Wilson 2020)
* Do the same using (30) and (40) to generate a hierarchy of ODEs
for $Z^{(k)}$ and $I^{(k)}$, respectively. (Wilson 2020)
* Solve for $\phi$ in terms of $\theta_i$, (13), (14) (Park and Wilson 2020)
* Compute the higher-order interaction functions (15) (Park and Wilson 2020)



Notes:
- <code>pA</code> requires endpoint=False. make sure corresponding <code>dxA</code>s are used.




    
## Functions


    
### Function `module_exists` {#StrongCoupling.module_exists}




>     def module_exists(
>         module_name
>     )





    
## Classes


    
### Class `StrongCoupling` {#StrongCoupling.StrongCoupling}




>     class StrongCoupling(
>         rhs,
>         coupling,
>         LC_init,
>         var_names,
>         pardict,
>         **kwargs
>     )


See the defaults dict below for allowed kwargs.

All model parameters must follow the convention
'parameter_val'. No other underscores should be used.
the script splits the parameter name at '_' and uses the
string to the left as the sympy parmeter name.

Reserved names: ...
    
    rhs: callable.
        right-hand side of a model
    coupling: callable.
        coupling function between oscillators
    LC_init: list or numpy array.
        initial condition of limit cycle (must be found manually).
        XPP is useful, otherwise integrate your RHS for various
        initial conditions for long times and extract an initial
        condition close to the limit cycle.
    var_names: list.
        list of variable names as strings
    pardict: dict.
        dictionary of parameter values. dict['par1_val'] = float.
        Make sure to use par_val format, where each parameter name is
        followed by _val.
    recompute_LC: bool.
        If True, recompute limit cycle. If false, load limit cycle if
        limit cycle data exists. Otherwise, compute. Default: False.
    recompute_monodromy: bool.
        If true, recompute kappa, the FLoquet multiplier using the
        monodromy matrix. If false, load kappa if data exists,
        otherwise compute. Default: False.
    recompute_g_sym: bool.
        If true, recompute the symbolic equations for g^k. If false,
        load the symbolic equations if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_g: bool.
        If true, recompute the ODEs for g^k. If false,
        load the data for g^k if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_het_sym: bool.
        If true, recompute the symbolic equations for z^k and i^k.
        If false, load the symbolic equations if they exist in
        storage. Otherwise, compute. Default: False.
    recompute_z: bool.
        If true, recompute the ODEs for z^k. If false,
        load the data for z^k if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_i: bool.
        If true, recompute the ODEs for i^k. If false,
        load the data for i^k if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_k_sym: bool.
        If true, recompute the symbolic equations for K^k. If false,
        load the symbolic equations if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_p_sym: bool.
        If true, recompute the symbolic equations for p^k. If false,
        load the symbolic equations if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_k_sym: bool.
        If true, recompute the symbolic equations for H^k. If false,
        load the symbolic equations if they exist in storage.
        Otherwise, compute. Default: False.
    recompute_h: bool.
        If true, recompute the H functions for H^k. If false,
        load the data equations if they exist in storage.
        Otherwise, compute. Default: False.
    g_forward: list or bool.
        If bool, integrate forwards or backwards
        when computing g^k. If list, integrate g^k forwards or
        backwards based on bool value g_forward[k].
        Default: False.
    z_forward: list or bool.
        Same idea as g_forward for PRCS. Default: False.
    i_forward: list or bool.
        Same idea as g_forward for IRCS. Default: False.
    dense: bool.
        If True, solve_ivp uses dense=True and evaluate solution
        along tLC.
    dir: str.
        Location of data directory. Please choose carefully
        because some outputs may be on the order of gigabytes
        if NA >= 5000. Write 'home+data_dir/' to save to the folder
        'data_dir' in the home directory. Otherwise the script
        will use the current working directory by default uless
        an absolute path is used. The trailing '/' is
        required. Default: None.
    trunc_order: int.
        Highest order to truncate the expansion. For example, 
        trunc_order = 3 means the code will compute up to and 
        including order 3. Default: 3.
    NA: int.
        Number of partitions to discretize phase when computing p.
        Default: 500.
    p_iter: int.
        Number of periods to integrate when computing the time 
        interal in p. Default: 10.
    max_iter: int.
        Number of Newton iterations. Default: 20.
    TN: int.
        Total time steps when computing g, z, i.
    rtol, atol: float.
        Relative and absolute tolerance for ODE solvers.
        Defaults: 1e-7, 1e-7.
    rel_tol: float.
        Threshold for use in Newton scheme. Default: 1e-6.
    method: string.
        Specify the method used in scipy.integrate.solve_ivp.
        Default: LSODA.
    g_bad_dx: list or bool. If bool, use another variable to increase
        the magnitude of the Newton derivative. This can only be
        determined after attempting to run simulations and seeing that
        the Jacobian for the Newton step is ill-conditioned. If list,
        check for ill-conditioning for each order k.
        For example, we use g_small_dx = [False,True,False,...,False]
        for the thalamic model. The CGL model only needs
        g_small_idx = False
    z_Gbad_idx: same idea as g_small_idx for PRCs
    i_bad_idx: same idea as g_small_idx for IRCs







    
#### Methods


    
##### Method `bispeu` {#StrongCoupling.StrongCoupling.bispeu}




>     def bispeu(
>         self,
>         fn,
>         x,
>         y
>     )


silly workaround
<https://stackoverflow.com/questions/47087109/>        evaluate-the-output-from-scipy-2d-interpolation-along-a-curve

    
##### Method `fLam2` {#StrongCoupling.StrongCoupling.fLam2}




>     def fLam2(
>         self,
>         fn
>     )


interp2db object

    
##### Method `fmod` {#StrongCoupling.StrongCoupling.fmod}




>     def fmod(
>         self,
>         fn
>     )


fn has mod built-in

input function-like. usually interp1d object

needed to keep lambda input variable unique to fn.

otherwise lambda will use the same input variable for 
all lambda functions.

    
##### Method `generate_expansions` {#StrongCoupling.StrongCoupling.generate_expansions}




>     def generate_expansions(
>         self
>     )


generate expansions from Wilson 2020

    
##### Method `generate_g` {#StrongCoupling.StrongCoupling.generate_g}




>     def generate_g(
>         self,
>         k,
>         het_vec
>     )


generate Floquet eigenfunctions g

uses Newtons method

    
##### Method `generate_h_odd` {#StrongCoupling.StrongCoupling.generate_h_odd}




>     def generate_h_odd(
>         self,
>         k
>     )


interaction functions

note to self: see nb page 130 for notes on indexing in sums.
need to sum over to index N-1 out of size N to avoid
double counting boundaries in mod operator.

    
##### Method `generate_het_sym` {#StrongCoupling.StrongCoupling.generate_het_sym}




>     def generate_het_sym(
>         self
>     )


Generate heterogeneous terms for integrating the Z_i and I_i terms.

###### Returns

None.

    
##### Method `generate_i` {#StrongCoupling.StrongCoupling.generate_i}




>     def generate_i(
>         self,
>         k,
>         het_vec
>     )


i0 equation is stable in forwards time
i1, i2, etc equations are stable in backwards time.

    
##### Method `generate_k_sym` {#StrongCoupling.StrongCoupling.generate_k_sym}




>     def generate_k_sym(
>         self
>     )




    
##### Method `generate_limit_cycle` {#StrongCoupling.StrongCoupling.generate_limit_cycle}




>     def generate_limit_cycle(
>         self
>     )




    
##### Method `generate_p` {#StrongCoupling.StrongCoupling.generate_p}




>     def generate_p(
>         self,
>         k
>     )




    
##### Method `generate_p_old` {#StrongCoupling.StrongCoupling.generate_p_old}




>     def generate_p_old(
>         self,
>         k
>     )




    
##### Method `generate_z` {#StrongCoupling.StrongCoupling.generate_z}




>     def generate_z(
>         self,
>         k,
>         het_vec
>     )




    
##### Method `interp_lam` {#StrongCoupling.StrongCoupling.interp_lam}




>     def interp_lam(
>         self,
>         k,
>         fn_dict,
>         fn_type='z'
>     )


it is too slow to call individual interpolated functions
in the symbolic heterogeneous terms.
soince the heterogeneous terms only depend on t, just make
and interpolated version and use that instead so only 1 function
is called for the het. terms per iteration in numerical iteration.

    
##### Method `load_g` {#StrongCoupling.StrongCoupling.load_g}




>     def load_g(
>         self
>     )


load all Floquet eigenfunctions g or recompute

    
##### Method `load_g_sym` {#StrongCoupling.StrongCoupling.load_g_sym}




>     def load_g_sym(
>         self
>     )




    
##### Method `load_h` {#StrongCoupling.StrongCoupling.load_h}




>     def load_h(
>         self
>     )




    
##### Method `load_h_sym` {#StrongCoupling.StrongCoupling.load_h_sym}




>     def load_h_sym(
>         self
>     )


also compute h lam

    
##### Method `load_het_sym` {#StrongCoupling.StrongCoupling.load_het_sym}




>     def load_het_sym(
>         self
>     )




    
##### Method `load_i` {#StrongCoupling.StrongCoupling.load_i}




>     def load_i(
>         self
>     )


load all IRCs i or recomptue

    
##### Method `load_k_sym` {#StrongCoupling.StrongCoupling.load_k_sym}




>     def load_k_sym(
>         self
>     )


kA, kB contain the ith order terms of expanding the coupling fun.
cA, cB contain the derivatives of the coupling fn.

    
##### Method `load_limit_cycle` {#StrongCoupling.StrongCoupling.load_limit_cycle}




>     def load_limit_cycle(
>         self
>     )




    
##### Method `load_monodromy` {#StrongCoupling.StrongCoupling.load_monodromy}




>     def load_monodromy(
>         self
>     )


if monodromy data exists, load. if DNE or 
recompute required, compute here.

    
##### Method `load_p` {#StrongCoupling.StrongCoupling.load_p}




>     def load_p(
>         self
>     )


generate/load the ODEs for psi.

    
##### Method `load_p_sym` {#StrongCoupling.StrongCoupling.load_p_sym}




>     def load_p_sym(
>         self
>     )


generate/load the het. terms for psi ODEs.
    
to be solved using integrating factor meothod.

pA['sym'][k] is the forcing function of order k

    
##### Method `load_z` {#StrongCoupling.StrongCoupling.load_z}




>     def load_z(
>         self
>     )


load all PRCs z or recompute

    
##### Method `monodromy` {#StrongCoupling.StrongCoupling.monodromy}




>     def monodromy(
>         self,
>         t,
>         z
>     )


calculate right-hand side of system


$\dot \Phi = J\Phi, \Phi(0)=I$,

where $\Phi$ is a matrix solution

jacLC is the jacobian evaluated along the limit cycle

    
##### Method `numerical_jac` {#StrongCoupling.StrongCoupling.numerical_jac}




>     def numerical_jac(
>         self,
>         fn,
>         x,
>         eps=1e-07
>     )


return numerical Jacobian function



    
# Module `Thalamic` {#Thalamic}

Example: Thalamic model from Wilson and Ermentrout RSTA 2019,
Rubin and Terman JCNS 2004




    
## Functions


    
### Function `coupling` {#Thalamic.coupling}




>     def coupling(
>         vars_pair,
>         pdict,
>         option='val'
>     )


Synaptic coupling function between Thalamic oscillators.

E.g.,this Python function is the function $G(x_i,x_j)$
in the equation
$\frac{dx_i}{dt} = F(x_i) + \varepsilon G(x_i,x_j)$

Parameters

    vars_pair : list or array
        contains state variables from oscillator A and B, e.g.,
        vA, hA, rA, wA, vB, hB, rB, wB  
    pdict : dict of flots or sympy objects.
        parameter dictionary pdict[key], val. key is always a string
        of the parameter. val is either the parameter value (float) or 
        the symbolic version of the parameter key.
    option : string.
        Set to 'val' when inputs, t, z, pdict are floats. Set to
        'sym' when inputs t, z, pdict are sympy objects. The default
        is 'val'.

Returns
    
    numpy array or sympy Matrix
        returns numpy array if option == 'val'. 
        returns sympy Matrix if option == 'sym'

    
### Function `main` {#Thalamic.main}




>     def main()




    
### Function `rhs` {#Thalamic.rhs}




>     def rhs(
>         t,
>         z,
>         pdict,
>         option='val'
>     )


Right-hand side of the Thalamic model from Wilson and Ermentrout
RSTA 2019 and Rubin and Terman JCNS 2004


Parameters

    t : float or sympy object.
        time
    z : array or list of floats or sympy objects.
        state variables of the thalamic model v, h, r, w.
    pdict : dict of flots or sympy objects.
        parameter dictionary pdict[key], val. key is always a string
        of the parameter. val is either the parameter value (float) or 
        the symbolic version of the parameter key.
    option : string.
        Set to 'val' when inputs, t, z, pdict are floats. Set to
        'sym' when inputs t, z, pdict are sympy objects. The default
        is 'val'.
    
Returns
    
    numpy array or sympy Matrix
        returns numpy array if option == 'val'
        returns sympy Matrix if option == 'sym'




    
# Module `CGL` {#CGL}

Example: Complex Ginzburgh-Landau (CGL) model from Wilson and Ermentrout RSTA
2019




    
## Functions


    
### Function `coupling` {#CGL.coupling}




>     def coupling(
>         vars_pair,
>         pdict,
>         option='value'
>     )


Diffusive coupling function between Complex Ginzburgh Landau
(CGL) oscillators.

E.g.,this Python function is the function $G(x_i,x_j)$
in the equation
$\frac{dx_i}{dt} = F(x_i) + \varepsilon G(x_i,x_j)$

Parameters

    vars_pair : list or array
        contains state variables from oscillator A and B, e.g.,
        x1,y1,x2,y2
    pdict : dict of flots or sympy objects.
        parameter dictionary pdict[key], val. key is always a string
        of the parameter. val is either the parameter value (float) or 
        the symbolic version of the parameter key.
    option : string.
        Set to 'val' when inputs, t, z, pdict are floats. Set to
        'sym' when inputs t, z, pdict are sympy objects. The default
        is 'val'.

Returns
    
    * numpy array or sympy Matrix
        * returns numpy array if option == 'val'. 
        returns sympy Matrix if option == 'sym'

    
### Function `main` {#CGL.main}




>     def main()




    
### Function `rhs` {#CGL.rhs}




>     def rhs(
>         t,
>         z,
>         pdict,
>         option='value'
>     )


Right-hand side of the Complex Ginzburgh-Landau (CGL) model from
Wilson and Ermentrout RSTA 2019 

Parameters
    
    t : float or sympy object.
        time
    z : array or list of floats or sympy objects.
        state variables of the thalamic model v, h, r, w.
    pdict : dict of flots or sympy objects.
        parameter dictionary pdict[key], val. key is always a string
        of the parameter. val is either the parameter value (float) or 
        the symbolic version of the parameter key.
    option : string.
        Set to 'val' when inputs, t, z, pdict are floats. Set to
        'sym' when inputs t, z, pdict are sympy objects. The default
        is 'val'.
    
Returns
    
    numpy array or sympy Matrix
        returns numpy array if option == 'val'
        returns sympy Matrix if option == 'sym'




Generated by *pdoc* 0.9.1 (<https://pdoc3.github.io>).
