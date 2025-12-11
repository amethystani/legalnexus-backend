#!/bin/bash
# LaTeX Setup Script for macOS
# This script will install LaTeX (BasicTeX) and required packages

set -e

echo "========================================="
echo "LaTeX Setup Script for macOS"
echo "========================================="
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew is not installed."
    echo "Please install Homebrew first: https://brew.sh"
    exit 1
fi

echo "✓ Homebrew found"

# Check if LaTeX is already installed
if command -v pdflatex &> /dev/null; then
    echo "✓ LaTeX (pdflatex) is already installed"
    PDFLATEX_VERSION=$(pdflatex --version | head -n 1)
    echo "  Version: $PDFLATEX_VERSION"
else
    echo ""
    echo "LaTeX is not installed. Installing BasicTeX..."
    echo "This will require your admin password (sudo)."
    echo ""
    
    # Install BasicTeX
    brew install --cask basictex
    
    # Update PATH
    eval "$(/usr/libexec/path_helper)"
    
    # Add to PATH for current session
    export PATH="/Library/TeX/texbin:$PATH"
    
    echo ""
    echo "✓ BasicTeX installed"
fi

# Verify installation
if ! command -v pdflatex &> /dev/null; then
    # Try common locations
    if [ -f "/Library/TeX/texbin/pdflatex" ]; then
        export PATH="/Library/TeX/texbin:$PATH"
        echo "✓ Found pdflatex in /Library/TeX/texbin"
    else
        echo "Error: pdflatex still not found after installation."
        echo "Please restart your terminal and run this script again."
        exit 1
    fi
fi

echo ""
echo "Installing required LaTeX packages..."
echo "This may take a few minutes..."

# Update tlmgr (TeX Live Manager)
sudo tlmgr update --self

# Install required packages for the presentation
PACKAGES=(
    "beamer"
    "tikz"
    "pgf"
    "amsmath"
    "algorithmicx"
    "algorithm2e"
    "xcolor"
    "graphicx"
    "multirow"
    "amssymb"
    "collection-fontsrecommended"
)

for package in "${PACKAGES[@]}"; do
    echo "  Installing $package..."
    sudo tlmgr install "$package" || echo "  Warning: $package installation had issues (may already be installed)"
done

echo ""
echo "========================================="
echo "✓ LaTeX setup complete!"
echo "========================================="
echo ""
echo "You can now compile your presentation with:"
echo "  ./compile_presentation.sh"
echo ""
echo "Or manually with:"
echo "  pdflatex presentation.tex"
echo "  pdflatex presentation.tex"
echo ""


