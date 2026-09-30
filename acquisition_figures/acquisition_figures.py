#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:28:29 2026

@author: dmzj518
"""

"""
searchmap_position_viewer.py

Workflow
--------
1. Read BatchPositionsList.xml
2. Store Position XYZ coordinates
3. Compute Position acquisition bounding boxes
4. Read all SearchMaps
5. Compute SearchMap bounding boxes
6. Match each Position to a SearchMap
7. Read Tile XMLs
8. Determine which Tile contains each Position
9. Convert stage coordinates to SearchMap pixels
10. Draw acquisition rectangle
11. Display SearchMap + acquired image
"""

from pathlib import Path
from dataclasses import dataclass
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import mrcfile
import re



###############################################################################
# SETTINGS
###############################################################################

DATA_DIR = Path("")

PROCESS = "ARETOMO"  #Write the method of motion correction and processing:ARETOMO or WARP (all caps)

THUMBNAIL_ROOT = Path("") #path to the warp_tiltseries/tiltstack directory. Ignore if using ARETOMO3 or Windows Warp

ARETOMO_DIRECTORY = Path("") #directory with Aretomo stacks. If using Linux Warp ignore, but if uisng Windows Warp 


# YOU CAN KEEP THESE AS THEY ARE

POSITION_XML_DIR=DATA_DIR

BATCH_XML = DATA_DIR / "Batch" / "BatchPositionsList.xml"

SEARCHMAP_DIR = DATA_DIR / "SearchMaps"


OUTPUT_DIR = DATA_DIR / "Acquisition_figure"

OUTPUT_DIR.mkdir(exist_ok=True)



###############################################################################
# DATA CLASSES
###############################################################################

@dataclass
class BoundingBox:

    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class Position:

    name: str

    x: float
    y: float
    z: float

    tileset_name: str

    status: str
    defocus: float

    # Acquired image metadata
    pixel_size: float = None
    width: int = None
    height: int = None

    image_path: Path = None
    xml_path: Path = None

    bbox: BoundingBox = None

    searchmap: object = None
    tile: object = None


@dataclass
class Tile:

    # XML filename
    filename: str

    # Stage coordinates (meters)
    x: float
    y: float
    z: float

    # Image calibration
    pixel_size: float       # meters/pixel

    # Image dimensions
    width: int              # pixels
    height: int             # pixels
    xml_path: Path

    # Relationship between image pixels and stage coordinates
    transform: np.ndarray = None

    # Physical footprint in stage coordinates
    bbox: BoundingBox = None

    # Optional: actual image file
    image_path: Path = None
    
    jpg_path: Path = None
    jpg_width: int = None
    jpg_height: int = None


@dataclass
class SearchMap:

    name: str

    x: float
    y: float
    z: float

    pixel_size: float
    width: int
    height: int

    # stage/image relationship


    image_path: Path
    xml_path: Path
    transform: np.ndarray = None
    bbox: BoundingBox = None
    tiles: list = None


###############################################################################
# XML PARSERS
###############################################################################

def find_position_thumbnail(position_name, thumbnail_root, process_name, aretomo_dir=None):
    """
    Find the first acquired image for a position.

    Selects files matching:
        Position_X_001_*.png

    """
    if process_name=="WARP":
        print("looking for thumbnails in")
        position_dir = (
            thumbnail_root /
            position_name /
            "thumbnails"
            )
        
        if not position_dir.exists():
            print(
                f"No thumbnail directory for {position_name}. Have you run WarpTools ts_stack succesfully?"
            )
            return None
        
        else:
            files = list(position_dir.glob(f"{position_name}_001_*.png"))
        
    if process_name=="ARETOMO":
        position_dir = Path("{}/{}.mrc".format(aretomo_dir, position_name))
        png_path = "{}/{}_001.png".format(aretomo_dir, position_name)
        if os.path.isfile(png_path)==False:
            print("Thumbnail for {} doesn't exist. Creating now".format(position_name))
            if os.path.isfile(position_dir)==False:
                
                print("No {}.mrc in directory, trying searching for other possible filenames with .mrc or .st extension".format(position_name))
                pattern = re.compile(
                    rf"^{re.escape(position_name)}(?:\.(?:mrc|st)|_[A-Za-z][A-Za-z0-9_-]*\.(?:mrc|st))$"
                )
                
                matches = [
                    p for p in Path(aretomo_dir).iterdir()
                    if p.is_file() and pattern.match(p.name)
                ]
                if len(matches)==0:
                    print("No files associated with {} found".format(position_name))
                    return None
                
                print("Found potential stack files: {}".format(matches))
                
                for p in matches:
                    try:
                        with mrcfile.open(p, permissive=False) as mrc:
                            print(f"{p} is a valid MRC file, using to create a thumbnail")
                            position_dir = p
                            break
                    except Exception:
                        print(f"{p} is not a valid MRC file")
            
            with mrcfile.open(position_dir, permissive=True) as mrc:
                tilt_no=len(mrc.data)
                print("tilt number", tilt_no)
                tilt_0=math.ceil(tilt_no/2)
                print("tilt 0 is slice", tilt_0)
                image = mrc.data[tilt_0]
            
            plt.imsave(
                png_path,
                image,
                cmap="gray")
            
            print("thumbnail saved to", png_path)
            
        files = list(png_path)
        
    if len(files) == 0:
        print(
            f"No 001 image found for {position_name}"
        )
        return None

    return files[0]

def load_batch_positions(xml_file, thumbnail_root, process="WARP", aretomo_dir=None):
    """
    Parse BatchPositionsList.xml

    Returns
    -------
    list[Position]
    """

    ns = {
        "bp": "Applications.Tomography.BatchPositionsList.Version.1"
    }

    positions = []

    tree = ET.parse(xml_file)
    root = tree.getroot()

    for entry in root.findall(".//bp:BatchPositionParameters", ns):

        name = entry.findtext("bp:Name", namespaces=ns)

        status = entry.findtext("bp:BatchPositionStatus", namespaces=ns)

        defocus = float(
            entry.findtext("bp:Defocus", namespaces=ns)
        )

        # Actual stage position
        x = float(
            entry.findtext("bp:StagePositionX", namespaces=ns)
        )

        y = float(
            entry.findtext("bp:StagePositionY", namespaces=ns)
        )

        z = float(
            entry.findtext("bp:StagePositionZ", namespaces=ns)
        )

        # SearchMap information
        position_on_tileset = entry.find(
            "bp:PositionOnTileSet",
            ns
        )

        tileset_name = None

        if position_on_tileset is not None:
            tileset_name = position_on_tileset.findtext(
                "bp:TileSetName",
                namespaces=ns
            )

        position = Position(
            name=name,
            x=x,
            y=y,
            z=z,
            tileset_name=tileset_name,
            status=status,
            defocus=defocus,
            image_path=find_position_thumbnail(name,thumbnail_root,process,aretomo_dir))
        
        
        if position.image_path is not None:
            positions.append(position)
    
        
        extra = entry.find("bp:AdditionalExposureTemplateAreas", ns)

        if extra is not None:
        
            for exposure in extra.findall("bp:ExposureTemplateAreaParameters", ns):
        
                dx = float(exposure.findtext("bp:PositionX", namespaces=ns))
                dy = float(exposure.findtext("bp:PositionY", namespaces=ns))
        
                position = Position(
                    name=exposure.findtext("bp:Name", namespaces=ns),
    
                    # offset from parent stage position
                    x=x - dx,
                    y=y - dy,
                    z=z,
    
                    tileset_name=tileset_name,
                    status=status,
                    defocus=float(
                        exposure.findtext("bp:Defocus", namespaces=ns)
                    ),
    
                    image_path=find_position_thumbnail(
                        exposure.findtext("bp:Name", namespaces=ns),
                        thumbnail_root,process, aretomo_dir))
                
                print(position.name)
                
                if position.image_path is not None:
                    positions.append(position)
    return positions



def load_position_metadata(position_dir, positions):
    """
    Load metadata from the first acquired image XML
    for each Position.

    Only uses files matching:
        Position_<N>_001_*.xml

    Parameters
    ----------
    position_dir : Path
        Directory containing Position XML files

    positions : list[Position]

    Returns
    -------
    list[Position]
    """

    # ns = {
    #     "m": "http://schemas.datacontract.org/2004/07/Fei.SharedObjects",
    #     "a": "http://schemas.datacontract.org/2004/07/System.Drawing"
    # }


    position_dict = {
        p.name: p
        for p in positions
    }


    for xml in position_dir.glob("Position_*_001_*.xml"):


        #
        # Extract Position name
        #
        # Example:
        # Position_10_001_-9.00_20260715_220029_Fractions
        #
        # parts:
        # ["Position", "10", "001", "-9.00", ...]
        #

        stem = xml.stem

        position_name = stem.split("_001_", 1)[0]


        if position_name not in position_dict:
            print(
                "No batch entry for",
                position_name
            )
            continue


        position = position_dict[position_name]


        tree = ET.parse(xml)
        root = tree.getroot()


        # -----------------------------
        # Detector image size
        # -----------------------------
        
        width = root.findtext(
            ".//ImageSize/Width"
        )
        
        height = root.findtext(
            ".//ImageSize/Height"
        )
        
        
        if width is not None:
            width = int(width)
        
        if height is not None:
            height = int(height)
        
        
        
        # -----------------------------
        # Pixel size
        # -----------------------------
        
        pixel_size_x = root.findtext(
            ".//SensorPixelSize/Width"
        )
        
        pixel_size_y = root.findtext(
            ".//SensorPixelSize/Height"
        )
        
        
        if pixel_size_x is not None:
            pixel_size_x = float(pixel_size_x)
        
        if pixel_size_y is not None:
            pixel_size_y = float(pixel_size_y)
       
        # use X pixel size
        position.pixel_size = pixel_size_x
        

        if width is not None:
            position.width = int(width)

        if height is not None:
            position.height = int(height)


        
        # Save XML and image paths
        #
        position.xml_path = xml

        #
        # Create footprint
        #
        if (
            position.pixel_size is not None
            and position.width is not None
            and position.height is not None
        ):

            position.bbox = create_bbox(
                position.x,
                position.y,
                position.pixel_size,
                position.width,
                position.height
            )


    return positions

def load_searchmaps(searchmap_dir):
    """
    Read every SearchMap directory.

    Returns
    -------
    list[SearchMap]
    """

    ns = {
        "m": "http://schemas.datacontract.org/2004/07/Fei.SharedObjects",
        "a": "http://schemas.datacontract.org/2004/07/System.Drawing"
    }


    searchmaps = []


    for folder in searchmap_dir.iterdir():

        if not folder.is_dir():
            continue


        xml = folder / "SearchMap.xml"

        if not xml.exists():
            continue


        # print("Reading", xml)


        tree = ET.parse(xml)
        root = tree.getroot()


        #
        # Stage coordinates
        #
        x = root.findtext(
            ".//m:microscopeData/"
            "m:stage/"
            "m:Position/"
            "m:X",
            namespaces=ns
        )

        y = root.findtext(
            ".//m:microscopeData/"
            "m:stage/"
            "m:Position/"
            "m:Y",
            namespaces=ns
        )

        z = root.findtext(
            ".//m:microscopeData/"
            "m:stage/"
            "m:Position/"
            "m:Z",
            namespaces=ns
        )


        #
        # Pixel size (meters/pixel)
        #
        pixel_size = root.findtext(
            ".//m:SpatialScale/"
            "m:pixelSize/"
            "m:x/"
            "m:numericValue",
            namespaces=ns
        )


        #
        # Image dimensions
        #
        width = root.findtext(
            ".//m:microscopeData/"
            "m:acquisition/"
            "m:camera/"
            "m:ReadoutArea/"
            "a:width",
            namespaces=ns
        )

        height = root.findtext(
            ".//m:microscopeData/"
            "m:acquisition/"
            "m:camera/"
            "m:ReadoutArea/"
            "a:height",
            namespaces=ns
        )


        #
        # Convert values
        #
        x = float(x) if x else None
        y = float(y) if y else None
        z = float(z) if z else None

        pixel_size = (
            float(pixel_size)
            if pixel_size
            else None
        )

        width = (
            int(width)
            if width
            else None
        )

        height = (
            int(height)
            if height
            else None
        )


        #
        # Create object
        #
        sm = SearchMap(

            name=folder.name,

            x=x,
            y=y,
            z=z,

            pixel_size=pixel_size,

            width=width,
            height=height,

            image_path=folder / "SearchMap.tif",

            xml_path=xml,

            bbox=None,

            tiles=[]
        )


        #
        # Create SearchMap footprint
        #
        if all([
            sm.pixel_size,
            sm.width,
            sm.height
        ]):

            sm.bbox = create_bbox(
                sm.x,
                sm.y,
                sm.pixel_size,
                sm.width,
                sm.height
            )

        searchmaps.append(sm)


    return searchmaps

def load_tiles(searchmap):
    """
    Read every Tile_*.xml inside a SearchMap directory.

    Parameters
    ----------
    searchmap : SearchMap

    Returns
    -------
    list[Tile]
    """

    ns = {
        "m": "http://schemas.datacontract.org/2004/07/Fei.SharedObjects",
        "a": "http://schemas.datacontract.org/2004/07/System.Drawing"
    }

    from PIL import Image

    tiles = []

    tile_folder = searchmap.xml_path.parent


    for xml in tile_folder.glob("Tile*.xml"):
        # print(xml)

        tree = ET.parse(xml)
        root = tree.getroot()


        # -----------------------------
        # Stage position
        # -----------------------------

        stage = root.find(
            ".//m:microscopeData/m:stage/m:Position",
            ns
        )

        if xml.name =="TileSetAcquisitionOptions.xml":
            continue
        
        if stage is None:
            print(f"No stage position in {xml.name}")
            continue


        x = float(stage.findtext("m:X", namespaces=ns))
        y = float(stage.findtext("m:Y", namespaces=ns))
        z = float(stage.findtext("m:Z", namespaces=ns))


        # -----------------------------
        # Pixel size
        # -----------------------------

        pixel_size = root.findtext(
            ".//m:SpatialScale/m:pixelSize/m:x/m:numericValue",
            namespaces=ns
        )

        if pixel_size is None:
            print(f"No pixel size in {xml.name}")
            continue

        pixel_size = float(pixel_size)


        # -----------------------------
        # Image dimensions
        # -----------------------------

        width = root.findtext(
            ".//m:microscopeData/m:acquisition/"
            "m:camera/m:ReadoutArea/a:width",
            namespaces=ns
        )

        height = root.findtext(
            ".//m:microscopeData/m:acquisition/"
            "m:camera/m:ReadoutArea/a:height",
            namespaces=ns
        )

        if width is None or height is None:
            print(f"No dimensions in {xml.name}")
            continue


        width = int(width)
        height = int(height)


        # -----------------------------
        # JPEG image
        # -----------------------------

        jpg = xml.with_suffix(".jpg")
        
        

        jpg_width = None
        jpg_height = None

        if jpg.exists():

            with Image.open(jpg) as img:
                jpg_width, jpg_height = img.size


        # -----------------------------
        # Bounding box
        # -----------------------------

        bbox = create_bbox(
            x,
            y,
            pixel_size,
            width,
            height
        )


        # -----------------------------
        # Create Tile
        # -----------------------------

        tile = Tile(

            filename=xml.name,

            x=x,
            y=y,
            z=z,

            pixel_size=pixel_size,

            width=width,
            height=height,

            xml_path=xml,

            image_path=jpg if jpg.exists() else None,

            jpg_width=jpg_width,
            jpg_height=jpg_height,

            bbox=bbox
        )


        tiles.append(tile)
    return tiles

###############################################################################
# GEOMETRY
###############################################################################

def create_bbox(center_x,
                center_y,
                pixel_size,
                width_px,
                height_px):
    """
    Compute microscope footprint.
    """

    width = width_px * pixel_size
    height = height_px * pixel_size

    return BoundingBox(
        xmin=center_x - width / 2,
        xmax=center_x + width / 2,
        ymin=center_y - height / 2,
        ymax=center_y + height / 2
    )


def point_inside_bbox(x, y, bbox):

    return (
        bbox.xmin <= x <= bbox.xmax
        and
        bbox.ymin <= y <= bbox.ymax
    )


# ###############################################################################
# # MATCHING
# ###############################################################################

def match_searchmap(position, searchmaps):
    print(position.name, "is within", position.tileset_name)
    
    sm = next(s for s in searchmaps if s.name == position.tileset_name)
    
    # if point_inside_bbox(position.x,
    #                          position.y,
    #                          sm.bbox):
    #     print("The stage position information in", position.name, "xml files matches the stage position in", sm.name, "xml files")
    #     print("The acquisition area will be plotted")
    #     return sm
    # else:
    #     print("The stage position information in", position.name, "xml files does not match the stage position in", sm.name, "xml files")
    #     print("The acquisition area will have to be checked manually")
    #     return sm
    return sm

    # return None


def match_tile(position, tiles):

    for tile in tiles:
        # print(
    #         tile.filename,
    #         "x:", tile.bbox.xmin, "→", tile.bbox.xmax,
    #         "y:", tile.bbox.ymin, "→", tile.bbox.ymax)
    # print("position:", position.x, position.y)

        if point_inside_bbox(position.x,
                             position.y,
                             tile.bbox):

            return tile

    return None


# ###############################################################################
# # COORDINATE CONVERSION
# ###############################################################################

def stage_to_tile_pixel(position,
                        tile):
    """
    Convert acquisition stage coordinates
    into Tile JPEG pixel coordinates.
    """


    #
    # Position relative to tile centre
    #
    dx = position.x - tile.x
    dy = position.y - tile.y


    #
    # Convert metres -> Tile MRC pixels
    #
    mrc_px = (
        dx / tile.pixel_size
        + tile.width / 2
    )

    mrc_py = (
        dy / tile.pixel_size
        + tile.height / 2
    )


    #
    # Convert MRC pixels -> JPEG pixels
    #
    scale_x = (
        tile.jpg_width /
        tile.width
    )

    scale_y = (
        tile.jpg_height /
        tile.height
    )


    jpg_px = mrc_px * scale_x
    jpg_py = mrc_py * scale_y


    return jpg_px, jpg_py



# ###############################################################################
# # PLOTTING
# ###############################################################################

def make_figure(position,
                searchmap,
                tile):
    """
    Create side-by-side figure.
    """

    fig_width_cm = 9.04 + 11.86 +2
    fig_height_cm = 11.86 +2
    
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(
            fig_width_cm / 2.54,
            fig_height_cm / 2.54
        ),
        gridspec_kw={
            "width_ratios": [9.04, 11.86]
        }
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.98,
        wspace=0.05
    )
    
    for a in ax:
        a.set_anchor("S")   # South = bottom
        a.axis("off")

    ###################################################################
    # Tile image
    ###################################################################

    if tile.image_path is not None and tile.image_path.exists():

        img = plt.imread(tile.image_path)
    
        ax[0].imshow(
            img,
            cmap="gray"
        )
    
    
        #
        # Convert stage coordinates to tile JPEG pixels
        #
        px, py = stage_to_tile_pixel(
            position,
            tile
        )
    
        py = tile.jpg_height - py

        
        # Acquisition footprint in physical units
        # (from Position class)
        #
        footprint_width_m = (
            position.width *
            position.pixel_size
        )
    
        footprint_height_m = (
            position.height *
            position.pixel_size
        )
    
    
        #
        # Convert physical footprint to tile pixels
        #
        width_tile_px = (
            footprint_width_m /
            tile.pixel_size
        )
    
        height_tile_px = (
            footprint_height_m /
            tile.pixel_size
        )
    
    
        #
        # Convert tile pixels to JPEG pixels
        #
        rectangle_width = (
            width_tile_px *
            tile.jpg_width /
            tile.width
        )
    
        rectangle_height = (
            height_tile_px *
            tile.jpg_height /
            tile.height
        )
    
    
        from matplotlib.patches import Rectangle


        rect = Rectangle(
            (
                px - rectangle_width / 2,
                py - rectangle_height / 2
                ),
            rectangle_width,
            rectangle_height,
            edgecolor="red",
            facecolor="none",
            linewidth=0.8
            )


        ax[0].add_patch(rect)


        ax[0].plot(
            px,
            py,
            "r+",
            markersize=5,
            markeredgewidth=0.8
            )


    else:

        ax[0].text(
            0.5,
            0.5,
            "No tile image found",
            ha="center",
            va="center",
            transform=ax[0].transAxes
    )


    ax[0].set_title(position.name)
###################################################################
# Position image
###################################################################

    if position.image_path is not None and position.image_path.exists():

        img = plt.imread(
            position.image_path
            )

        ax[1].imshow(
            img,
            cmap="gray"
            )

    else:

        ax[1].text(
            0.5,
            0.5,
            "No position image found",
            ha="center",
            va="center",
            transform=ax[1].transAxes
            )

    #ax[1].set_title(position.name)
    
    ax[0].axis("off")
    ax[1].axis("off")
    ax[0].set_aspect("equal")
    ax[1].set_aspect("equal")

    plt.tight_layout()

    outfile = OUTPUT_DIR / f"{position.name}.png"

    plt.savefig(
        outfile,
        dpi=200,
        bbox_inches="tight"
        )

    plt.close(fig)


###############################################################################
# MAIN
###############################################################################

# def main():

print("Loading Positions...")
print("Process run:", PROCESS)

positions = load_batch_positions(BATCH_XML, THUMBNAIL_ROOT, PROCESS, ARETOMO_DIRECTORY)

print("Loading Position metadata...")

positions = load_position_metadata(POSITION_XML_DIR, positions)


# Create acquisition footprint for every position
for p in positions:

    p.bbox = create_bbox(
        p.x,
        p.y,
        p.pixel_size,
        p.width,
        p.height
    )

print(f"Loaded {len(positions)} positions")


print("Loading SearchMaps...")

searchmaps = load_searchmaps(SEARCHMAP_DIR)

print(f"Loaded {len(searchmaps)} SearchMaps")


# Load tiles belonging to each SearchMap
for sm in searchmaps:

    sm.tiles = load_tiles(sm)

    print(
        sm.name,
        f"({len(sm.tiles)} tiles)"
    )


print("Matching Positions...")


for position in positions:


    # First find which SearchMap contains this position
    sm = match_searchmap(
        position,
        searchmaps
    )


    if sm is None:

        print(
            position.name,
            "No SearchMap found"
        )

        continue


    # Then find the specific tile
    tile = match_tile(
        position,
        sm.tiles
    )


    if tile is None:

        print(
            position.name,
            "No Tile found in",
            sm.name
        )

        continue


    print(
        position.name,
        "->",
        sm.name,
        "->",
        tile.filename
    )


    # Store the matches for later plotting
    position.searchmap = sm
    position.tile = tile

    make_figure(
        position,
        sm,
        tile
    )

    print(position.name, "done")

print("Finished.")


# if __name__ == "__main__":
#     main()