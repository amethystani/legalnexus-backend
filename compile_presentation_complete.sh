#!/bin/bash
# Complete LaTeX compilation script - tries Docker first, then local LaTeX

set -e

echo "========================================="
echo "Compiling presentation.tex to PDF"
echo "========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Method 1: Try Docker (if available and running)
if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
    echo "Method 1: Using Docker (no local LaTeX needed)"
    echo ""
    
    LATEX_IMAGE="blang/latex:ctanfull"
    
    echo "Pulling LaTeX Docker image..."
    docker pull "$LATEX_IMAGE" > /dev/null 2>&1 || docker pull "$LATEX_IMAGE"
    
    echo "Compiling..."
    docker run --rm \
        -v "$SCRIPT_DIR":/workspace \
        -w /workspace \
        "$LATEX_IMAGE" \
        sh -c "pdflatex -interaction=nonstopmode -halt-on-error presentation.tex && pdflatex -interaction=nonstopmode -halt-on-error presentation.tex"
    
    if [ -f "presentation.pdf" ]; then
        echo ""
        echo "========================================="
        echo "✓ SUCCESS! PDF generated: presentation.pdf"
        echo "========================================="
        ls -lh presentation.pdf
        exit 0
    fi
fi

# Method 2: Try local LaTeX installation
echo ""
echo "Method 2: Using local LaTeX installation"
echo ""

# Find pdflatex
PDFLATEX=""
if command -v pdflatex &> /dev/null; then
    PDFLATEX="pdflatex"
elif [ -f "/Library/TeX/texbin/pdflatex" ]; then
    PDFLATEX="/Library/TeX/texbin/pdflatex"
    export PATH="/Library/TeX/texbin:$PATH"
elif [ -f "/usr/local/texlive/2024/bin/universal-darwin/pdflatex" ]; then
    PDFLATEX="/usr/local/texlive/2024/bin/universal-darwin/pdflatex"
    export PATH="/usr/local/texlive/2024/bin/universal-darwin:$PATH"
elif [ -f "/usr/local/texlive/2025/bin/universal-darwin/pdflatex" ]; then
    PDFLATEX="/usr/local/texlive/2025/bin/universal-darwin/pdflatex"
    export PATH="/usr/local/texlive/2025/bin/universal-darwin:$PATH"
fi

if [ -n "$PDFLATEX" ]; then
    echo "Using: $PDFLATEX"
    echo ""
    echo "First compilation pass..."
    $PDFLATEX -interaction=nonstopmode -halt-on-error presentation.tex
    
    echo ""
    echo "Second compilation pass..."
    $PDFLATEX -interaction=nonstopmode -halt-on-error presentation.tex
    
    if [ -f "presentation.pdf" ]; then
        echo ""
        echo "========================================="
        echo "✓ SUCCESS! PDF generated: presentation.pdf"
        echo "========================================="
        ls -lh presentation.pdf
        exit 0
    fi
fi

# If we get here, neither method worked
echo ""
echo "========================================="
echo "Error: Could not compile presentation"
echo "========================================="
echo ""
echo "LaTeX is not installed and Docker is not available."
echo ""
echo "To install LaTeX, run:"
echo "  brew install --cask basictex"
echo "  eval \"\$(/usr/libexec/path_helper)\""
echo "  sudo tlmgr update --self"
echo "  sudo tlmgr install beamer tikz pgf amsmath algorithmicx algorithm2e xcolor graphicx multirow amssymb collection-fontsrecommended"
echo ""
echo "Or start Docker Desktop and run this script again."
echo ""
exit 1


