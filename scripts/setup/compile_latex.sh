#!/bin/bash
# LaTeX Compilation Script for collegereport.tex

# Try to find pdflatex in common locations
if command -v pdflatex &> /dev/null; then
    PDFLATEX="pdflatex"
elif [ -f "/Library/TeX/texbin/pdflatex" ]; then
    PDFLATEX="/Library/TeX/texbin/pdflatex"
    export PATH="/Library/TeX/texbin:$PATH"
elif [ -f "/usr/local/texlive/2024/bin/universal-darwin/pdflatex" ]; then
    PDFLATEX="/usr/local/texlive/2024/bin/universal-darwin/pdflatex"
    export PATH="/usr/local/texlive/2024/bin/universal-darwin:$PATH"
else
    echo "Error: pdflatex not found. Please install LaTeX first:"
    echo "  Option 1: brew install --cask basictex"
    echo "  Option 2: Download MacTeX from https://www.tug.org/mactex/"
    echo ""
    echo "After installation, run: eval \"\$(/usr/libexec/path_helper)\""
    exit 1
fi

# Check for biber (for bibliography)
if command -v biber &> /dev/null; then
    BIBER="biber"
elif [ -f "/Library/TeX/texbin/biber" ]; then
    BIBER="/Library/TeX/texbin/biber"
    export PATH="/Library/TeX/texbin:$PATH"
else
    echo "Warning: biber not found. Bibliography may not compile correctly."
    BIBER=""
fi

echo "Using pdflatex: $PDFLATEX"
if [ -n "$BIBER" ]; then
    echo "Using biber: $BIBER"
fi

# Compilation process
echo ""
echo "Step 1: First pdflatex pass..."
$PDFLATEX -interaction=nonstopmode collegereport.tex

if [ -n "$BIBER" ]; then
    echo ""
    echo "Step 2: Running biber for bibliography..."
    $BIBER collegereport
fi

echo ""
echo "Step 3: Second pdflatex pass..."
$PDFLATEX -interaction=nonstopmode collegereport.tex

echo ""
echo "Step 4: Third pdflatex pass (for cross-references)..."
$PDFLATEX -interaction=nonstopmode collegereport.tex

echo ""
if [ -f "collegereport.pdf" ]; then
    echo "✓ Success! PDF generated: collegereport.pdf"
    ls -lh collegereport.pdf
else
    echo "✗ Error: PDF not generated. Check collegereport.log for errors."
    exit 1
fi


