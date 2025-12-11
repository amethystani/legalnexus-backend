#!/usr/bin/env python3
"""
Properly reorganize presentation slides
"""
import re

# Read file
with open('presentation.tex', 'r') as f:
    lines = f.readlines()

# Find frame boundaries
frames = {}
current_frame = None
current_lines = []
in_frame = False

for i, line in enumerate(lines):
    if '\\begin{frame}' in line:
        # Extract frame title
        match = re.search(r'\\begin{frame}(?:\[.*?\])?\{(.+?)\}', line)
        if match:
            current_frame = match.group(1)
            current_lines = [line]
            in_frame = True
    elif in_frame:
        current_lines.append(line)
        if '\\end{frame}' in line:
            frames[current_frame] = current_lines
            in_frame = False
            current_frame = None
            current_lines = []

# Get preamble (everything before first real frame after title)
preamble_lines = []
for i, line in enumerate(lines):
    if '\\frame{\\titlepage}' in line:
        # Include up to and including titlepage
        preamble_lines = lines[:i+2]  # +2 to include the blank line after
        break

# Define desired order
desired_order = [
    # Introduction
    "Problem: Why Legal AI is Hard",
    "What Others Are Doing (Related Work)",
    
    # Methodology  
    "Our Solution: Three Novel Components",
    "Component 1: Why Hyperbolic Space?",
    "Poincaré Ball Visualization",
    "Hyperbolic Training",
    "Hyperbolic vs Euclidean: Visual Comparison",
    "Component 2: Multi-Agent Swarm",
    "What is Nash Equilibrium?",
    "Nash Equilibrium Formulation",
    "Component 3: Adversarial Hybrid Retrieval",
    "Prosecutor-Defense-Judge Simulation",
    "System Architecture Overview",
    "Why Our Method is Novel",
    
    # Dataset
    "Dataset \\& Technology",
    
    # RESULTS SECTION
    "HGCN Results",
    "Multi-Agent Results",
    "Overall Performance",
    "Performance Metrics",
    "Comparison",
    "Evaluation Methodology",
    
    # Examples
    "Real Example: Concrete Query",
    "Use Case 1: Legal Research",
    "Use Case 2: Judicial Decision Support",
    "Use Case 3: Access to Justice",
    
    # Conclusion
    "Limitations \\& Challenges",
    "Broader Impact",
    "Future Directions",
    "Conclusion",
    "Thank You!",
]

# Build new file
new_lines = preamble_lines.copy()

# Add frames in desired order
added = set()
for title in desired_order:
    if title in frames:
        new_lines.extend(frames[title])
        new_lines.append('\n')
        added.add(title)
        print(f"✓ Added: {title}")
    else:
        print(f"✗ Not found: {title}")

# Add any missing frames
for title, frame_lines in frames.items():
    if title not in added:
        new_lines.extend(frame_lines)
        new_lines.append('\n')
        print(f"⚠ Added extra: {title}")

# Add document end
new_lines.append('\\end{document}\n')

# Write
with open('presentation.tex', 'w') as f:
    f.writelines(new_lines)

print(f"\n✓ Reorganized {len(frames)} frames")
print("\n📊 RESULTS SECTION is now grouped together (slides 16-21):")
print("  16. HGCN Results")
print("  17. Multi-Agent Results")  
print("  18. Overall Performance")
print("  19. Performance Metrics")
print("  20. Comparison")
print("  21. Evaluation Methodology")
