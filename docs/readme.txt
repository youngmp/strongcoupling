Last run on Ubuntu 18.04.5
pdoc3 0.9.1
pandoc 2.7.3
pdoc3 and pandoc installed using pip3

To create docs, cd to project home, run the commands

$ pdoc3 --pdf --template-dir ./docs/ StrongCoupling Thalamic CGL > docs/docstrings.md

This command reads all the docstrings from the files StrongCoupling.py, Thalamic.py, and CGL.py and puts them into markdown format in docs/docstrings.md

To convert the raw md file to tex and to include the intoductory text in README.md, cd to the docs directory and run the command

$ pandoc --from=markdown+abbreviations+tex_math_single_backslash --toc --toc-depth=4 --output=docs/docs.tex -t latex -s README.md docs/docstrings.md 

README.md contains custom information generated outside of docstrings (introduction, recommended versions). Make sure to put docstrings.md last because pdoc3 includes a footnote that disrupts any md text that comes after it

-s flag enables standalone, so it generates the .tex file that can be compiled directly using PDFLaTeX.
