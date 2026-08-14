#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 12:46:51 2026

@author: dmzj518
"""

from pptx import Presentation
from pptx.util import Inches
from pathlib import Path
import re

### PATHS

template = "path/to/acquisition_summary_template.pptx"
output = "path/to/acquisition/presentation.pptx"
image_folder= Path("path/to/directory/with/saved/images")
legend="path/to/acquisition_legend.png"

def sort_key(path):
    match = re.fullmatch(
        r"Position_(\d+)(?:_(\d+))?",
        path.stem,
    )

    if not match:
        return (float("inf"), float("inf"))

    position = int(match.group(1))
    subposition = int(match.group(2)) if match.group(2) is not None else -1

    return (position, subposition)


def find_shape(slide, name):
    
    for shape in slide.placeholders:
        # print('%d %s' % (shape.placeholder_format.idx, shape.name))
        if shape.name == name:
            return shape

    raise ValueError(f"Shape '{name}' not found")


def replace_with_image(slide, placeholder_name, image_path):
    placeholder = find_shape(slide, placeholder_name)

    slide.shapes.add_picture(
        str(image_path),
        placeholder.left,
        placeholder.top,
        placeholder.width,
        placeholder.height,
    )

    sp = placeholder._element
    sp.getparent().remove(sp)

prs = Presentation(template)

images = sorted(
    (
        image
        for image in image_folder.iterdir()
        if image.is_file()
        and image.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ),
    key=sort_key,
)

for image in images:
    print("inserting", image.name)

    if not image.is_file():
        continue

    if image.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
        continue

    # Copy the template slide
    slide_layout = prs.slide_layouts[0]
    slide = prs.slides.add_slide(slide_layout)

    replace_with_image(
        slide,
        "ClipArt Placeholder 2",
        image)
    replace_with_image(slide, "ClipArt Placeholder 3", legend)

prs.save(output)
    