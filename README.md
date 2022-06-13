# simple-imzml-to-ometiff
Simple imzML to OME-TIFF for HuBMAP data ingestion.

## Usage

Install git and miniconda if you don't already have a conda capable set up.

1. Clone repository: 
`git clone https://github.com/NHPatterson/simple-imzml-to-ometiff`
2. Create new conda environment: `conda create -n imzml-conv python=3.8`
3. Activate environment: `conda activate imzml-conv`
4. Move terminal working directory to cloned GitHub repository: `cd simple-imzml-to-ometiff`
5. Install requirements using `requirements.txt`: `pip install -r requirements.txt`
6. Use script (outlined below)

### Running converter

This will convert the `.imzML` to OME-TIFF for peaks in `peaks.csv`.
The first argument is always the `.imzML` filepath and the second is to the `peaks.csv`.
The third argument is the mass tolerance over which to integrate the peak.

Using explicit pixel spacing:
```bash
python imzml_to_ometiff.py "path/to/imzml/file.imzML" "path/to/peaks.csv" 0.2 --x_um 10 --y_um 10`
```

Using implicit pixel spacing (spacing is calucated by taking the distance in `horz/vert_um` / `n_pixels_x/y`)
```bash
python imzml_to_ometiff.py "path/to/imzml/file.imzML" "path/to/peaks.csv" 0.2 --horz_um 3000 --vert_um 4000`
```

Getting help:
```bash
python imzml_to_ometiff.py -h
```
```console
usage: imzml_to_ometiff.py [-h] [--x_um X_SPACING] [--y_um Y_SPACING]
                           [--horz_um HORIZONTAL_UM] [--vert_um VERTICAL_UM]
                           [--out_dir OUTPUT_DIR] [--mass_unit MASS_UNIT]
                           imzML peaks_csv mz_tol

Static conversion imzML to OME-TIFF for HuBMAP ingestion

positional arguments:
  imzML                 filepath to the imzML file
  peaks_csv             filepath to the peaks csv (must contain two columns:
                        'mz' & 'peak_name')
  mz_tol                peak width

optional arguments:
  -h, --help            show this help message and exit
  --x_um X_SPACING      spacing of the image in x dimension (horizontal),
                        provide this value or the total horizontal length in
                        micronsin --horz_um
  --y_um Y_SPACING      spacing of the image in y dimension (vertical),
                        provide this value or the total horizontal length in
                        microns in --vert_um
  --horz_um HORIZONTAL_UM
                        total distance in microns in the x dimension
                        (horizontal), physical spacing is calculated as total
                        distance (x) / number of pixels (x)
  --vert_um VERTICAL_UM
                        total distance in microns in the y dimension
                        (vertical), physical spacing is calculated as total
                        distance (y) / number of pixels (y)
  --out_dir OUTPUT_DIR  where to store output OME-TIFFdefaults to working
                        directory
  --mass_unit MASS_UNIT
                        unit of the mass element, i.e. 'm/z' or 'm', appears
                        before the mass value: PC(34:0) - m/z 734.58

```



### Preparing peaks.csv

This simple table contains two columns: "mass" and "peak_name". Mass will be the mass (m/z or otherwise) where the 
peak is detected. `peak_name` is where the name of the peak that will be retained in the OME-TIFF that will be displayed on the 
HuBMAP portal. Peak unit can be set through the command line, default is m/z.

Example:
`PC(34:0) - m/z 734.58`


