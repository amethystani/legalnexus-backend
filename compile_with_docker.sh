#!/bin/bash
# Compile LaTeX presentation using Docker (no local LaTeX installation needed)

set -e

echo "========================================="
echo "Compiling presentation.tex using Docker"
echo "========================================="
echo ""

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Error: Docker is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Check for required image files
echo "Checking for required image files..."
MISSING=0

if [ ! -f "poincare_3d_visualization.png" ]; then
    echo "  ⚠ Warning: poincare_3d_visualization.png not found"
    MISSING=1
else
    echo "  ✓ poincare_3d_visualization.png found"
fi

if [ ! -f "whatsapp_image.png" ]; then
    echo "  ⚠ Warning: whatsapp_image.png not found"
    MISSING=1
else
    echo "  ✓ whatsapp_image.png found"
fi

if [ ! -f "Shiv_Nadar_University_logo.png" ]; then
    echo "  ⚠ Info: Shiv_Nadar_University_logo.png not found (may not be used)"
else
    echo "  ✓ Shiv_Nadar_University_logo.png found"
fi

echo ""

# Use a LaTeX Docker image
LATEX_IMAGE="blang/latex:ctanfull"

echo "Pulling LaTeX Docker image (this may take a few minutes on first run)..."
docker pull "$LATEX_IMAGE" > /dev/null 2>&1 || {
    echo "Pulling image..."
    docker pull "$LATEX_IMAGE"
}

echo "✓ LaTeX image ready"
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Compiling presentation.tex..."
echo ""

# Compile using Docker
docker run --rm \
    -v "$SCRIPT_DIR":/workspace \
    -w /workspace \
    "$LATEX_IMAGE" \
    sh -c "pdflatex -interaction=nonstopmode -halt-on-error presentation.tex && pdflatex -interaction=nonstopmode -halt-on-error presentation.tex"

echo ""

if [ -f "presentation.pdf" ]; then
    echo "========================================="
    echo "✓ SUCCESS! PDF generated: presentation.pdf"
    echo "========================================="
    ls -lh presentation.pdf
    echo ""
    echo "PDF saved at: $SCRIPT_DIR/presentation.pdf"
else
    echo "✗ Error: PDF not generated."
    echo "Check for errors above or in presentation.log"
    exit 1
fi


