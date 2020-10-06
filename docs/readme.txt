To create docs, cd to project home, run the command

$ pdoc3 --pdf StrongCoupling > docs/docs.md

To compile the raw md file to PDF, run the command

$ pandoc --metadata=title:"StrongCoupling Documentation"               \
           --from=markdown+abbreviations+tex_math_single_backslash  \
           --pdf-engine=xelatex --variable=mainfont:"DejaVu Sans"   \
           --toc --toc-depth=4 --output=docs.pdf -t latex docs0.md docs.md

docs0.md contains custom information generated outside of docstrings (introduction, recommended versions), and docsmd contains information straight from the docstrings.
