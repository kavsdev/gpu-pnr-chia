## TikZ figures, typeset in the paper's own fonts (Libertine/newtxmath) at \small, drawn to \textwidth.
FIGS = fig_overview.pdf fig_protocol.pdf
all: $(FIGS)
%.pdf: %.tex common.tex
	pdflatex -interaction=nonstopmode -halt-on-error $<
clean:
	rm -f *.aux *.log
