---
description: |
    API documentation for modules: StrongCoupling, CGL, Thalamic.

lang: en

classoption: oneside
geometry: margin=1in
papersize: a4

linkcolor: blue
links-as-notes: true
...


    
# Module `StrongCoupling` {#StrongCoupling}

@author: Youngmin Park

The logical flow of the class follows the paper by Wilson 2020.
-produce heterogeneous terms for g for arbirary dx
-substitute dx with g=g0 + psi*g1 + psi^2*g2+...
-produce het. terms for irc
-

this file is also practice for creating a more general class for any RHS.

coupling functions for thalamic neurons from RTSA Ermentrout, Park, Wilson 2019


Notes
-----=
-PA requires endpoint=False. make sure corresponding dxAs are used.

TODO: add backwards as an option for i,g,z.




    
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


Thalamic model from RSTA 2019
Requires sympy, numpy, matplotlib.

See the defaults dict below for allowed kwargs.

All model parameters must follow the convention
'parameter_val'. No other underscores should be used.
the script splits the parameter name at '_' and uses the
string to the left as the sympy parmeter name.

Reserved names: ...

g_forward: list or bool. If bool, integrate forwards or backwards
    when computing g^k. If list, integrate g^k forwards or backwards
    based on bool value g_forward[k]
z_forward: list or bool. Same idea as g_forward for PRCS
i_forward: list or bool. Same idea as g_forward for IRCS

g_bad_dx: list or bool. If bool, use another variable to increase
    the magnitude of the Newton derivative. This can only be
    determined after attempting to run simulations and seeing that
    the Jacobian for the Newton step is ill-conditioned. If list,
    check for ill-conditioning for each order k.
    For example, we use g_small_dx = [False,True,False,...,False]
    for the thalamic model. The CGL model only needs
    g_small_idx = False
z_bad_idx: same idea as g_small_idx for PRCs
i_bad_idx: same idea as g_small_idx for IRCs

coupling_pars: str. example: input '_d='+str(d_par) to include the d
    parameter d_par in the hodd function name.







    
#### Methods


    
##### Method `bispeu` {#StrongCoupling.StrongCoupling.bispeu}




>     def bispeu(
>         self,
>         fn,
>         x,
>         y
>     )


silly workaround
<https://stackoverflow.com/questions/47087109/...>
evaluate-the-output-from-scipy-2d-interpolation-along-a-curve

    
##### Method `dg` {#StrongCoupling.StrongCoupling.dg}




>     def dg(
>         self,
>         t,
>         z,
>         order,
>         het_vec
>     )


g functon rhs with ith het. term

z: position
t: time
jacLC: jacobian on LC
het: heterogeneous terms

order determines the Taylor expansion term

    
##### Method `di` {#StrongCoupling.StrongCoupling.di}




>     def di(
>         self,
>         t,
>         z,
>         order,
>         het_vec
>     )


g functon rhs with ith het. term

z: position
t: time
jacLC: jacobian on LC
het: heterogeneous terms

order determines the Taylor expansion term

    
##### Method `dz` {#StrongCoupling.StrongCoupling.dz}




>     def dz(
>         self,
>         t,
>         z,
>         order,
>         het_vec
>     )


g functon rhs with ith het. term

z: position
t: time
jacLC: jacobian on LC
het: heterogeneous terms

order determines the Taylor expansion term

    
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



    
# Module `CGL` {#CGL}

The logical flow of the class follows the paper by Wilson 2020.
-produce heterogeneous terms for g for arbirary dx
-substitute dx with g=g0 + psi*g1 + psi^2*g2+...
-produce het. terms for irc
-

this file is also practice for creating a more general class for any RHS.


Todo
-----=
-make sure that np.dot and sym matrix products are consistent.
-check that np.identity and sym.eye are consistent




    
## Functions


    
### Function `coupling` {#CGL.coupling}




>     def coupling(
>         vars_pair,
>         pdict,
>         option='value'
>     )


r^(2n) to r^n function. default parameter order is from perspective of
first oscillator.

in this case the input is (x1,y1,x2,y2) and the output is an R^2 vec.

    
### Function `main` {#CGL.main}




>     def main()




    
### Function `rhs` {#CGL.rhs}




>     def rhs(
>         t,
>         z,
>         pdict,
>         option='value'
>     )


right-hand side of the equation of interest. CCGL model.

write in standard python notation as if it will be used in an ODE solver.

###### Returns

`right-hand side equauation in terms` of <code>the inputs. if x,y scalars,</code>
:   &nbsp;


return scalar. If x,y, sympy symbols, return symbol.




    
# Module `Thalamic` {#Thalamic}

file for comparing to CGL. implement adjoint methods in Wilson 2020

<https://stackoverflow.com/questions/49306092/parsing-a-symbolic-expression-that-includes-user-defined-functions-in-sympy>

user-defined




    
## Functions


    
### Function `coupling` {#Thalamic.coupling}




>     def coupling(
>         vars_pair,
>         pdict,
>         option='val'
>     )




    
### Function `main` {#Thalamic.main}




>     def main()




    
### Function `rhs` {#Thalamic.rhs}




>     def rhs(
>         t,
>         z,
>         pdict,
>         option='val'
>     )


right-hand side of the equation of interest. thalamic neural model.



-----
Generated by *pdoc* 0.9.1 (<https://pdoc3.github.io>).
