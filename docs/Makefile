SHELL := /bin/bash
SOURCEDIR = source
BUILDDIR = build

all: apidoc
	sphinx-build -M html "$(SOURCEDIR)" "$(BUILDDIR)" -W --keep-going

apidoc: clean-apidoc
	sphinx-apidoc \
      --private \
      --module-first \
      --maxdepth 10 \
      --force \
      --output-dir "$(SOURCEDIR)/api" \
      ../src/sipm_studio \
      ../src/sipm_studio/_version.py

clean-apidoc:
	rm -rf "$(SOURCEDIR)/api"

clean: clean-apidoc
	rm -rf "$(BUILDDIR)"

# # Minimal makefile for Sphinx documentation
# #

# # You can set these variables from the command line, and also
# # from the environment for the first two.
# SPHINXOPTS    ?=
# SPHINXBUILD   ?= sphinx-build
# SOURCEDIR     = source
# BUILDDIR      = build

# # Put it first so that "make" without argument is like "make help".
# help:
# 	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# .PHONY: help Makefile

# # Catch-all target: route all unknown targets to Sphinx using the new
# # "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
# %: Makefile
# 	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
