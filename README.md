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
