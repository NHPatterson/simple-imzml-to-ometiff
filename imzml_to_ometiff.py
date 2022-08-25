from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
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
    mass_unit: str = "m/z",
    imsml: Optional[Union[Path, str]] = None,
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

    if imsml:
        if Path(imsml).suffix.lower() == ".h5":
            with h5py.File(imsml, "r") as f:
                xy_orig = f["xy_original"][:]
                xy_pad = f["xy_padded"][:]
        elif Path(imsml).suffix.lower() == ".csv":
            xy_imsml = pd.read_csv(imsml)
            xy_pad = np.asarray(xy_imsml[["x_padded", "y_padded"]])
            xy_orig = np.asarray(xy_imsml[["x_original", "y_original"]])
        else:
            raise ValueError(
                f"{Path(imsml).suffix.lower()} is "
                f"not a recognized extension : [.h5,.csv]"
            )

        xy_imzml = np.asarray(imzml.coordinates)[:, [0, 1]]
        imzml_index = np.arange(0, len(imzml.coordinates), dtype=np.int64)

        df_xy_imzml = pd.DataFrame(
            np.column_stack([xy_imzml, imzml_index]),
            columns=["x_original", "y_original", "imzml_index"],
        )
        df_xy_pad = pd.DataFrame(
            np.column_stack([xy_orig, xy_pad]),
            columns=["x_original", "y_original", "x_pad", "y_pad"]
        )

        coordinate_df = pd.merge(df_xy_pad, df_xy_imzml)

        coordinate_df.sort_values("imzml_index", inplace=True)
        coordinate_df["z"] = 1

        new_coords = list(coordinate_df[["x_pad", "y_pad", "z"]].itertuples(index=False, name=None))
        imzml.coordinates = new_coords
        imzml.imzmldict["max count of pixels x"] = np.max(coordinate_df["x_pad"]) + 1
        imzml.imzmldict["max count of pixels y"] = np.max(coordinate_df["y_pad"]) + 1


    out_file = Path(output_dir) / f"{Path(imzml_file).stem}.ome.tiff"

    # parse ion images
    ion_images = []
    for mz in tqdm(peak_mzs, desc="ion images processed"):
        ii = getionimage(imzml, mz, tol=mz_tolerance, reduce_func=np.sum)
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
        help="where to store output OME-TIFF defaults to working directory",
        default="./",
    )

    parser.add_argument(
        "--imsml",
        dest="imsml",
        type=str,
        help="IMS MicroLink registered coordinates",
        default=None,
    )

    parser.add_argument(
        "--mass_unit",
        dest="mass_unit",
        type=str,
        help="unit of the mass element, i.e. 'm/z' or 'm', appears before the mass value: "
        " PC(34:0) - m/z 734.58",
    )

    parser.set_defaults(
        output_dir="./",
        x_spacing=None,
        y_spacing=None,
        vertical_um=None,
        horizontal_um=None,
        imsml=None,
    )

    args = parser.parse_args()
    print(args)
    peak_data = pd.read_csv(args.peaks_csv)

    if "mass" not in peak_data.columns:
        raise ValueError(f"no mass column in peak data, columns: {peak_data.columns}")

    if "peak_name" not in peak_data.columns:
        raise ValueError(f"no name column in peak data, columns: {peak_data.columns}")

    imzml_to_ometiff(
        args.imzml,
        np.asarray(peak_data["mass"]),
        np.asarray(peak_data["peak_name"]),
        horizontal_um=args.horizontal_um,
        vertical_um=args.vertical_um,
        x_spacing=args.x_spacing,
        y_spacing=args.y_spacing,
        output_dir=args.output_dir,
        mz_tolerance=args.mz_tol,
        mass_unit=args.mass_unit,
        imsml=args.imsml,
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())

