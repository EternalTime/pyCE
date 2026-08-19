# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
# -W turns Sphinx warnings into errors, so a stale toctree entry, a malformed
# directive or a duplicate label fails the build instead of scrolling past.
# (Unresolved Python cross-references stay silent unless -n is added too.)
# It lives here rather than in the CI command so that the plain
# `make -C docs html` published in the README behaves the same on a
# contributor's machine as it does on GitHub.  `?=` keeps it overridable, e.g.
# `SPHINXOPTS= make html` while drafting.
SPHINXOPTS   ?= -W
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = pyCE
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)