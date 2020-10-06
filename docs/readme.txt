Last run on Ubuntu 18.04.5
pdoc3 0.9.1
pandoc 2.7.3
pdoc3 and pandoc installed using pip3

To create docs, cd to project home, run the command

$ pdoc3 --pdf StrongCoupling > docs/docs.md

To compile the raw md file to PDF, run the command

$ pandoc --metadata=title:"StrongCoupling Documentation"           \
         --from=markdown+abbreviations+tex_math_single_backslash   \
         --toc --toc-depth=4 --output=docs.tex -t latex -s docs0.md docs.md

docs0.md contains custom information generated outside of docstrings (introduction, recommended versions), and docs.md contains information straight from the docstrings.

-s flag enables standalone, so it generates the .tex file that can be compiled directly using PDFLaTeX.
