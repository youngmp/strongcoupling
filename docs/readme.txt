Last run on Ubuntu 18.04.5
pdoc3 0.9.1
pandoc 2.7.3
pdoc3 and pandoc installed using pip3

To create docs, cd to project home, run the command

$ pdoc3 --pdf --template-dir ./docs/ StrongCoupling Thalamic CGL > docs/docstrings.md

This command reads all the docstrings from the files StrongCoupling.py, Thalamic.py, and CGL.py and puts them into markdown format in docs/docstrings.md

To combine the README.md and docs/docstrings.md files into a single .tex file, run the command (from project home)

$ pandoc --from=markdown+abbreviations+tex_math_single_backslash --toc --toc-depth=4 --output=docs/docs.tex -t latex -s README.md docs/docstrings.md 

-s flag enables standalone, so it generates the .tex file that can be compiled directly using PDFLaTeX.

README.md contains custom information generated outside of docstrings (introduction, recommended versions).