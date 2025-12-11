#!/bin/bash
# Install required LaTeX packages

export PATH="/Library/TeX/texbin:$PATH"

echo "Installing required LaTeX packages..."
echo "This will require your admin password."

sudo tlmgr update --self
sudo tlmgr install algorithm algorithmicx

echo ""
echo "Packages installed! Now you can compile the presentation."


