# 

<p align="center">
  <img src="https://huggingface.co/datasets/JulioContrerasH/DataMLSTAC/resolve/main/banner_satcube.png" width="50%">
</p>

<p align="center">
    <em>A Python package for managing Sentinel-2 satellite data cubes</em> 🚀
</p>

<p align="center">
<a href='https://pypi.python.org/pypi/satcube'>
    <img src='https://img.shields.io/pypi/v/satcube.svg' alt='PyPI' />
</a>
<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://pycqa.github.io/isort/" target="_blank">
    <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="isort">
</a>
</p>

---

**GitHub**: [https://github.com/IPL-UV/satcube](https://github.com/IPL-UV/satcube) 🌐

**PyPI**: [https://pypi.org/project/satcube/](https://pypi.org/project/satcube/) 🛠️

---

## **Overview** 📊

**satcube** is a Python package designed for efficient management, processing, and analysis of Sentinel-2 satellite image cubes. It allows for downloading, cloud masking, gap filling, and super-resolving Sentinel-2 imagery, as well as creating monthly composites and performing interpolation.

## **Key Features** ✨
- **Satellite image download**: Retrieve Sentinel-2 images from Earth Engine efficiently. 🛰️
- **Cloud masking**: Automatically remove clouds from Sentinel-2 images. ☁️
- **Gap filling**: Fill missing data using methods like linear interpolation and histogram matching. 🧩
- **Super-resolution**: Apply super-resolution models to enhance image quality. 🔍
- **Monthly composites**: Aggregate images into monthly composites with various statistical methods. 📅
- **Temporal smoothing**: Smooth reflectance values across time using interpolation techniques. 📈
## **Installation** ⚙️

Install the latest version from PyPI:

```bash
pip install satcube
```

## **How to use** 🛠️

### **Basic usage: working with sentinel-2 data** 🌍

#### **Load libraries**

```python
import ee
import satcube
```

#### **Authenticate and initialize earth engine**

```python
ee.Authenticate()
ee.Initialize(project="ee-csaybar-real")
```
#### **Download model weights**
```python
outpath = satcube.download_weights(path="weights")
```

#### **Create a satellite dataCube**
```python
datacube = satcube.SatCube(
    coordinates=(-77.68598590138802,-8.888223962022263),
    sensor=satcube.Sentinel2(weight_path=outpath, edge_size=384),
    output_dir="wendy01",
    max_workers=12,
    device="cuda",
)
```


### **Query and process sentinel-2 data** 🛰️

#### **Query the sentinel-2 image collection**

```python
# Query the Sentinel-2 image collection
table_query = datacube.metadata_s2()

# Filter images based on cloud cover and remove duplicates
table_query_subset = table_query[table_query["cs_cdf"] > 0.30]
table_query_subset = table_query_subset.drop_duplicates(subset="img_date")
mgrs_tile_max = table_query_subset["mgrs_title"].value_counts().idxmax()
table_query_subset = table_query_subset[table_query_subset["mgrs_title"] == mgrs_tile_max]
```

#### **Download sentinel-2 images**

```python
table_download = datacube.download_s2_image(table_query_subset)
```
#### **Cloud masking**

```python
# Remove clouds from the images
table_nocloud = datacube.cloudmasking_s2(table_download)
table_nocloud = table_nocloud[table_nocloud["cloud_cover"] < 0.75]
table_nocloud.reset_index(drop=True, inplace=True)
```

#### **Gap filling**

```python
# Fill missing data in the images
table_nogaps = datacube.gapfilling_s2(table_nocloud)
table_nogaps = table_nogaps[table_nogaps["match_error"] < 0.1]
```
### **Monthly composites and image smoothing 📅**

#### **Create monthly composites**

```python
# Generate monthly composites
table_composites = datacube.monthly_composites_s2(
    table_nogaps, agg_method="median", date_range=("2016-01-01", "2024-07-31")
)
```

#### **Interpolate missing data**

```python
# Interpolate missing months if necessary
table_interpolate = datacube.interpolate_s2(table=table_composites)
```

#### **Smooth reflectance values**

```python
# Smooth reflectance values across time
table_smooth = datacube.smooth_s2(table=table_interpolate)
```

### **Super-resolution and visualization** 📐



#### **Super-resolution**

```python
# Apply super-resolution to the image cube
# table_final = datacube.super_s2(table_smooth)
```


#### **Display images**

```python
# Display the images from the data cube
datacube.display_images(table=table_smooth)
```

#### **Create a GIF**

```python
# !apt-get install imagemagick
import os
os.system("convert -delay 20 -loop 0 wendy01/z_s2_07_smoothed_png/temp_07*.png animation.gif")

from IPython.display import Image
Image(filename='animation.gif', width=500)
```

<p align="center">
  <img src="https://huggingface.co/datasets/JulioContrerasH/DataMLSTAC/resolve/main/gif_satcube.gif" width="100%">
</p>

#### **Smooth reflectance values**

```python
# Smooth reflectance values across time
table_smooth = datacube.smooth_s2(table=table_interpolate)
```

## **Supported features and filters** ✨

- **Cloud masking:** Efficient removal of clouds from satellite images.
- **Resampling methods:** Various methods for resampling and aligning imagery.
- **Super-resolution:** ONNX-based models for improving image resolution.