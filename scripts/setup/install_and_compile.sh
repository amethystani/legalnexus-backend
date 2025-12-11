#!/bin/bash
# Complete LaTeX Installation and Compilation Script
# Run this script - it will guide you through installation if needed

set -e

echo "========================================="
echo "LaTeX Installation & Compilation Script"
echo "========================================="
echo ""

# Function to check if LaTeX is installed
check_latex() {
    if command -v pdflatex &> /dev/null; then
        return 0
    elif [ -f "/Library/TeX/texbin/pdflatex" ]; then
        export PATH="/Library/TeX/texbin:$PATH"
        return 0
    else
        return 1
    fi
}

# Check if LaTeX is installed
if check_latex; then
    echo "✓ LaTeX is already installed"
    PDFLATEX_VERSION=$(pdflatex --version | head -n 1)
    echo "  $PDFLATEX_VERSION"
    echo ""
    SKIP_INSTALL=true
else
    echo "LaTeX is not installed."
    echo ""
    echo "To install LaTeX, please run this command in your terminal:"
    echo ""
    echo "  brew install --cask basictex"
    echo ""
    echo "After installation, run:"
    echo "  eval \"\$(/usr/libexec/path_helper)\""
    echo ""
    echo "Then install required packages:"
    echo "  sudo tlmgr update --self"
    echo "  sudo tlmgr install beamer tikz pgf amsmath algorithmicx algorithm2e xcolor graphicx multirow amssymb collection-fontsrecommended"
    echo ""
    echo "After installation, run this script again to compile the presentation."
    echo ""
    exit 1
fi

# Now compile the presentation
echo "Compiling presentation.tex..."
echo ""

cd "$(dirname "$0")"

# Find pdflatex
if command -v pdflatex &> /dev/null; then
    PDFLATEX="pdflatex"
elif [ -f "/Library/TeX/texbin/pdflatex" ]; then
    PDFLATEX="/Library/TeX/texbin/pdflatex"
    export PATH="/Library/TeX/texbin:$PATH"
fi

echo "Using: $PDFLATEX"
echo ""

# Check for required images
echo "Checking image files..."
if [ ! -f "poincare_3d_visualization.png" ]; then
    echo "  ⚠ Warning: poincare_3d_visualization.png not found"
fi
if [ ! -f "whatsapp_image.png" ]; then
    echo "  ⚠ Warning: whatsapp_image.png not found"
fi
echo ""

# Compile
echo "First compilation pass..."
$PDFLATEX -interaction=nonstopmode -halt-on-error presentation.tex

echo ""
echo "Second compilation pass..."
$PDFLATEX -interaction=nonstopmode -halt-on-error presentation.tex

echo ""
if [ -f "presentation.pdf" ]; then
    echo "========================================="
    echo "✓ SUCCESS! PDF generated: presentation.pdf"
    echo "========================================="
    ls -lh presentation.pdf
    echo ""
    echo "PDF location: $(pwd)/presentation.pdf"
else
    echo "✗ Error: PDF not generated. Check presentation.log"
    exit 1
fi


