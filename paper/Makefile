# ============================================================================
# Makefile for LaTeX Paper Compilation
# Paper: Linear Attention Architectures for Geopotential Height Forecasting
# ============================================================================

# Main document name (without .tex extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compiler flags
LATEXFLAGS = -interaction=nonstopmode -halt-on-error

# Output directory (optional, comment out to use current directory)
# OUTDIR = build

# ============================================================================
# Targets
# ============================================================================

.PHONY: all pdf bib clean cleanall view watch help

# Default target
all: pdf

# Build PDF with full compilation (latex -> bibtex -> latex -> latex)
pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib
	@echo "=== First LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "=== BibTeX pass ==="
	-$(BIBTEX) $(MAIN)
	@echo "=== Second LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "=== Third LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex
	@echo "=== Build complete: $(MAIN).pdf ==="

# Quick build (single pass, no bibliography update)
quick:
	@echo "=== Quick LaTeX pass ==="
	$(LATEX) $(LATEXFLAGS) $(MAIN).tex

# Bibliography only
bib:
	@echo "=== BibTeX pass ==="
	$(BIBTEX) $(MAIN)

# Clean auxiliary files
clean:
	@echo "=== Cleaning auxiliary files ==="
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out
	rm -f $(MAIN).toc $(MAIN).lof $(MAIN).lot $(MAIN).fls $(MAIN).fdb_latexmk
	rm -f $(MAIN).synctex.gz $(MAIN).nav $(MAIN).snm $(MAIN).vrb
	rm -f *.aux

# Clean everything including PDF
cleanall: clean
	@echo "=== Removing PDF ==="
	rm -f $(MAIN).pdf

# View PDF (Linux)
view: $(MAIN).pdf
	@if command -v xdg-open > /dev/null; then \
		xdg-open $(MAIN).pdf; \
	elif command -v evince > /dev/null; then \
		evince $(MAIN).pdf; \
	elif command -v okular > /dev/null; then \
		okular $(MAIN).pdf; \
	else \
		echo "No PDF viewer found. Please open $(MAIN).pdf manually."; \
	fi

# Watch for changes and recompile (requires inotifywait or latexmk)
watch:
	@if command -v latexmk > /dev/null; then \
		echo "=== Watching for changes (latexmk) ==="; \
		latexmk -pdf -pvc $(MAIN).tex; \
	elif command -v inotifywait > /dev/null; then \
		echo "=== Watching for changes (inotifywait) ==="; \
		while true; do \
			inotifywait -e modify $(MAIN).tex references.bib 2>/dev/null; \
			make pdf; \
		done; \
	else \
		echo "Error: Neither latexmk nor inotifywait found."; \
		echo "Install latexmk: sudo apt install latexmk"; \
		exit 1; \
	fi

# Use latexmk for smarter compilation
latexmk:
	@echo "=== Building with latexmk ==="
	latexmk -pdf $(MAIN).tex

# Word count (approximate)
count:
	@echo "=== Word count (approximate) ==="
	@if command -v texcount > /dev/null; then \
		texcount -inc -total $(MAIN).tex; \
	else \
		echo "texcount not found. Install: sudo apt install texlive-extra-utils"; \
		pdftotext $(MAIN).pdf - 2>/dev/null | wc -w || echo "pdftotext not available"; \
	fi

# Check for common issues
check:
	@echo "=== Checking for issues ==="
	@grep -n "\\\\remark" $(MAIN).tex && echo "WARNING: Found \\remark commands (TODO items)" || true
	@grep -n "TODO" $(MAIN).tex && echo "WARNING: Found TODO comments" || true
	@grep -n "\-\-" $(MAIN).tex | grep -v "^%" && echo "WARNING: Found -- (use endash or emdash)" || true

# Help
help:
	@echo "=============================================="
	@echo "LaTeX Paper Makefile"
	@echo "=============================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build PDF (default)"
	@echo "  pdf      - Build PDF with full compilation"
	@echo "  quick    - Quick build (single pass)"
	@echo "  bib      - Run BibTeX only"
	@echo "  clean    - Remove auxiliary files"
	@echo "  cleanall - Remove all generated files"
	@echo "  view     - Open PDF in viewer"
	@echo "  watch    - Watch for changes and recompile"
	@echo "  latexmk  - Build using latexmk"
	@echo "  count    - Word count"
	@echo "  check    - Check for common issues"
	@echo "  help     - Show this help"
	@echo ""
	@echo "Example workflow:"
	@echo "  make clean && make pdf && make view"
	@echo ""
