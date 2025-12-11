#!/usr/bin/env python3
"""
Reorganize presentation slides to group all results together
"""

# Read the presentation
with open('presentation.tex', 'r') as f:
    content = f.read()

# Define slide sections to extract and reorder
# We'll split content into sections and reorder them

# Split by frames
import re

# Find all frames with their content
frame_pattern = r'(\\begin{frame}(?:\[.*?\])?\{.*?\}.*?\\end{frame})'
frames = re.findall(frame_pattern, content, re.DOTALL)

# Get preamble (everything before first frame)
preamble_end = content.find('\\begin{frame}')
preamble = content[:preamble_end]

# Get title frame
title_frame_end = content.find('\\end{frame}', preamble_end) + len('\\end{frame}')
title_frame = content[preamble_end:title_frame_end]

# Now parse all other frames
remaining = content[title_frame_end:]
frames_with_names = []

# Extract frames with their titles
frame_starts = [m.start() for m in re.finditer(r'\\begin{frame}', remaining)]
frame_starts.append(len(remaining))

for i in range(len(frame_starts) - 1):
    frame_content = remaining[frame_starts[i]:frame_starts[i+1]]
    # Extract title
    title_match = re.search(r'\\begin{frame}(?:\[.*?\])?\{(.*?)\}', frame_content)
    if title_match:
        title = title_match.group(1)
        frames_with_names.append((title, frame_content))

# Define the new order
methodology_slides = [
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
]

results_slides = [
    "HGCN Results",
    "Multi-Agent Results",
    "Overall Performance",
    "Performance Metrics",
    "Comparison",
    "Evaluation Methodology",
]

# Build new content
new_content = preamble + title_frame + "\n"

# Add frames in new order
added_titles = set()

def add_frames_by_title(titles):
    global new_content, added_titles
    for title in titles:
        for frame_title, frame_content in frames_with_names:
            if title in frame_title and frame_title not in added_titles:
                new_content += frame_content
                added_titles.add(frame_title)
                break

# 1. Problem and Related Work
add_frames_by_title(["Problem: Why Legal AI is Hard", "What Others Are Doing (Related Work)"])

# 2. Methodology
add_frames_by_title(methodology_slides)

# 3. Dataset
add_frames_by_title(["Dataset & Technology"])  

# 4. ALL RESULTS
add_frames_by_title(results_slides)

# 5. Examples and Use Cases
add_frames_by_title([
    "Real Example: Concrete Query",
    "Use Case 1: Legal Research",
    "Use Case 2: Judicial Decision Support", 
    "Use Case 3: Access to Justice"
])

# 6. Conclusions
add_frames_by_title([
    "Limitations & Challenges",
    "Broader Impact",
    "Future Directions",
    "Conclusion",
    "Thank You!"
])

# Add any remaining frames not yet added
for frame_title, frame_content in frames_with_names:
    if frame_title not in added_titles:
        new_content += frame_content
        added_titles.add(frame_title)
        print(f"Added remaining frame: {frame_title}")

# Add document end
new_content += "\\end{document}\n"

# Write reorganized presentation
with open('presentation.tex', 'w') as f:
    f.write(new_content)

print("✓ Presentation reorganized!")
print(f"✓ Total frames reorganized: {len(added_titles)}")
print("\nNew slide order:")
for i, title in enumerate(added_titles, 1):
    print(f"{i}. {title}")
