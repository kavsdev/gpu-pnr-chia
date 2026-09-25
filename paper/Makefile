## Build the 4-page paper. Self-contained: acmart.cls + ACM-Reference-Format.bst
## live in this directory, so no system TeX install of acmart is needed.
## Usage: `make` (build), `make clean`, `make view`.

MAIN = main

$(MAIN).pdf: $(MAIN).tex references.bib acmart.cls ACM-Reference-Format.bst
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	-bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN).tex

.PHONY: clean view
clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out \
	      $(MAIN).pdf $(MAIN).fls $(MAIN).fdb_latexmk $(MAIN).toc

view: $(MAIN).pdf
	@echo "Built $(MAIN).pdf"
