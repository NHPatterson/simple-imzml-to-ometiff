from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
from pyimzml.ImzMLParser import ImzMLParser, getionimage
from tifffile import TiffWriter, OmeXml
from tqdm import tqdm


def imzml_to_ometiff(
    imzml_file: str,
    peak_mzs: List[float],
    peak_names: List[str],
    horizontal_um: Optional[Union[float, int]] = None,
    vertical_um: Optional[Union[float, int]] = None,
    x_spacing: Optional[Union[float, int]] = None,
    y_spacing: Optional[Union[float, int]] = None,
    output_dir: str = "./",
    mz_tolerance: float = 0.2,
    mass_unit: str = "m/z"
) -> str:
    """
    Function to read a processed Nano DESI imzML, extract peaks,
    and write a valid OME-TIFF with channel and spatial medata.
    Parameters
    ----------
    imzml_file: str
        Path to imzML file
    peak_mzs: list of float
        List of float values for m/z values to be placed in OME-TIFF
    peak_names: list of str
        Names for each m/z to be added as channel names in OME-TIFF
    horizontal_um: float or int:
        Total width of the acqusition, i.e., 18000 um
    vertical_um: float or int
        Total heighth of the acqusition, i.e., 6000 um
    output_dir: str or non
        Directory where OME-TIFF is to be written
    Returns
    -------
    out_file: str
        Path to newly created OME-TIFF
    """

    assert len(peak_names) == len(peak_mzs)

    imzml = ImzMLParser(imzml_file)

    out_file = Path(output_dir) / f"{Path(imzml_file).stem}.ome.tiff"

    # parse ion images
    ion_images = []
    for mz in tqdm(peak_mzs, desc="ion images processed"):
        ii = getionimage(imzml, mz_value=mz, tol=mz_tolerance)
        ion_images.append(ii)
    ion_images = np.stack(ion_images)

    # currently, HuBMAP portal supports float32
    ion_images = ion_images.astype(np.float32)

    # prepare OME-TIFF metadata
    channel_names = [f"{pn} - {mass_unit} {pz}" for pn, pz in zip(peak_names, peak_mzs)]

    n_ch = ion_images.shape[0]
    y_size = ion_images.shape[1]
    x_size = ion_images.shape[2]

    stored_shape = (n_ch, 1, 1, y_size, x_size, 1)
    im_shape = (n_ch, y_size, x_size)

    if vertical_um:
        phys_size_y = vertical_um / y_size
    elif y_spacing:
        phys_size_y = y_spacing
    else:
        raise ValueError(
            "Neither a vertical area nor a y spacing in microns was provided"
        )

    if horizontal_um:
        phys_size_x = horizontal_um / x_size
    elif x_spacing:
        phys_size_x = x_spacing
    else:
        raise ValueError(
            "Neither a horizontal area nor a x spacing in microns was provided"
        )

    ome_meta = {
        "PhysicalSizeX": phys_size_x,
        "PhysicalSizeY": phys_size_y,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeYUnit": "µm",
        "Name": Path(imzml_file).stem,
        "Channel": {"Name": channel_names},
    }

    omexml = OmeXml()
    omexml.addimage(
        dtype=ion_images.dtype,
        shape=im_shape,
        # specify how the image is stored in the TIFF file
        storedshape=stored_shape,
        **ome_meta,
    )

    ome_xml_str = omexml.tostring().encode("utf8")

    # write iamge page by page with OME metadata
    with TiffWriter(out_file, bigtiff=True) as tif:
        for channel_idx, image in enumerate(ion_images):
            options = dict(
                compression="deflate",
                photometric="minisblack",
                metadata=None,
            )
            # write OME-XML to the ImageDescription tag of the first page
            description = ome_xml_str if channel_idx == 0 else None
            # write channel data
            print(f" writing channel {channel_idx}")
            tif.write(
                image,
                description=description,
                **options,
            )

    return str(out_file)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Static conversion imzML to OME-TIFF for HuBMAP ingestion"
    )

    parser.add_argument(
        "imzml",
        metavar="imzML",
        type=str,
        help="filepath to the imzML file",
    )

    parser.add_argument(
        "peaks_csv",
        metavar="peaks_csv",
        type=str,
        help="filepath to the peaks csv (must contain two columns: 'mz' & 'peak_name')",
    )

    parser.add_argument(
        "mz_tol",
        metavar="mz_tol",
        type=float,
        help="peak width",
    )

    parser.add_argument(
        "--x_um",
        dest="x_spacing",
        type=float,
        help="spacing of the image in x dimension (horizontal), "
        "provide this value or the total horizontal length in microns"
        "in --horz_um",
    )

    parser.add_argument(
        "--y_um",
        dest="y_spacing",
        type=float,
        help="spacing of the image in y dimension (vertical), "
        "provide this value or the total horizontal length in microns "
        "in --vert_um",
    )

    parser.add_argument(
        "--horz_um",
        dest="horizontal_um",
        type=float,
        help="total distance in microns in the x dimension (horizontal), "
        "physical spacing is calculated as total distance (x) / number of pixels (x)",
    )

    parser.add_argument(
        "--vert_um",
        dest="vertical_um",
        type=float,
        help="total distance in microns in the y dimension (vertical), "
        "physical spacing is calculated as total distance (y) / number of pixels (y)",
    )

    parser.add_argument(
        "--out_dir",
        dest="output_dir",
        type=str,
        help="where to store output OME-TIFF" "defaults to working directory",
    )

    parser.add_argument(
        "--mass_unit",
        dest="mass_unit",
        type=str,
        help="unit of the mass element, i.e. 'm/z' or 'm', appears before the mass value: "
             " PC(34:0) - m/z 734.58"
    )

    parser.set_defaults(
        output_dir=None,
        x_spacing=None,
        y_spacing=None,
        vertical_um=None,
        horizontal_um=None
    )

    args = parser.parse_args()
    print(args)
    peak_data = pd.read_csv(args.peaks_csv)

    if "mass" not in peak_data.columns:
        raise ValueError(f"no mass column in peak data, columns: {peak_data.columns}")

    if "peak_name" not in peak_data.columns:
        raise ValueError(f"no name column in peak data, columns: {peak_data.columns}")

    imzml_to_ometiff(
        args.imzML,
        np.asarray(peak_data["mass"]),
        np.asarray(peak_data["peak_name"]),
        args.horizontal_um,
        args.vertical_um,
        args.x_spacing,
        args.y_spacing,
        args.output_dir,
        args.mz_tol,
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
