# LaTeX Setup Instructions for macOS

## Quick Installation (Recommended)

Since LaTeX installation requires admin privileges, please run this command in your terminal:

```bash
brew install --cask basictex
```

When prompted, enter your macOS admin password.

After installation completes, run:

```bash
eval "$(/usr/libexec/path_helper)"
```

Then install required packages:

```bash
sudo tlmgr update --self
sudo tlmgr install beamer tikz pgf amsmath algorithmicx algorithm2e xcolor graphicx multirow amssymb collection-fontsrecommended
```

## Alternative: Full MacTeX Installation

If you prefer the full LaTeX distribution (larger download ~4GB but includes everything):

1. Download from: https://www.tug.org/mactex/
2. Install the .pkg file
3. No additional packages needed

## Verify Installation

Check if LaTeX is installed:

```bash
pdflatex --version
```

## Compile Your Presentation

Once LaTeX is installed, compile your presentation:

```bash
./compile_presentation.sh
```

Or manually:

```bash
pdflatex presentation.tex
pdflatex presentation.tex
```

The PDF will be saved as `presentation.pdf` in the current directory.

## Troubleshooting

If `pdflatex` is not found after installation:

1. Restart your terminal
2. Or run: `eval "$(/usr/libexec/path_helper)"`
3. Or add to your `~/.zshrc`:
   ```bash
   export PATH="/Library/TeX/texbin:$PATH"
   ```


