call pandoc ^
 --latex-engine=xelatex ^
 -N ^
 --toc ^
 --toc-depth=4 ^
 --top-level-division=default ^
 --highlight-style=breezedark ^
 -V geometry:top=1in,bottom=1in,left=1in,right=1in,headheight=3ex,headsep=2ex ^
 -V CJKmainfont=SimSun ^
 --template=./pandoc_customized_template ^
 report_pandoc.md -o report_pandoc.pdf
