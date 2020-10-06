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

Python 3.7+ is necessary. Our code often requires more than 256 function inputs. Python 3.6 or earlier versions have a limit of 256 inputs and will not work with our scripts.

The script will likely work with earlier versions of all other libraries.

pathos is necessary for parallel computing and because pickling is more robust. I use lots of sympy objects that the multiprocessing library often can not pickle. In contrast, pathos uses dill, which can often pickle objects that the pickle module can not.

tqdm provides a status bar for parallel computing. It is not part of the engine, and the code can be modified to work without it. In future versions I may leave tqdm as a toggle.

## Installation

I will not release this script as an installable package simply because I do not have to time to maintain regular version releases for distribution platforms such as anaconda, pip, and apt. As long as your computer has the packages listed above and they are installed using Python 3.7, the script should run. Worst case scenario, yarrr matey a Windows 10 virutal machine and install everything using anaconda.

Note that in Ubuntu you may need to set up a virtual environment to be able to run the script. The system uses Python 3.6 by default and does not like it when the default is changed to Python 3.7.

-----
