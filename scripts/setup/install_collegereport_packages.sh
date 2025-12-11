#!/bin/bash
# Install required LaTeX packages for collegereport.tex

export PATH="/Library/TeX/texbin:$PATH"

echo "Installing required LaTeX packages for collegereport.tex..."
echo "This will require your admin password."

sudo tlmgr update --self
sudo tlmgr install xurl csquotes booktabs biblatex-chicago biber

echo ""
echo "Packages installed! Now you can compile collegereport.tex"


