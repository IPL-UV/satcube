import importlib.util
import pathlib
import shutil
from typing import Callable, List, Tuple, Union, Any

import ee
import fastcubo
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phicloudmask
import rasterio as rio
import requests
import satalign
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
import xarray as xr

from satcube.dataclass import Sentinel2


def metadata_s2(
    lon: float,
    lat: float,
    range_date: Tuple[str, str],
    edge_size: int,
    quiet: bool = False,
) -> pd.DataFrame:
    """Query the Sentinel-2 image collection.

    Args:
        lon (float): The longitude of the point.
        lat (float): The latitude of the point.
        range_date (Tuple[str, str]): The range of dates to query.
        edge_size (int): The edge size of the image.

    Returns:
        pd.DataFrame: The table with the images to download.
    """

    if not quiet:
        print(f"Querying Sentinel-2 image collection for {lon}, {lat}")

    # Query the image collection
    table = fastcubo.query_getPixels_imagecollection(
        point=(lon, lat),
        collection="COPERNICUS/S2_HARMONIZED",
        bands=[
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B10",
            "B11",
            "B12"
        ],  # We need all the bands to run cloud mask algorithms
        data_range=(range_date[0], range_date[1]),
        edge_size=edge_size,
        resolution=10,
    )

    # Add the cloud cover to the table
    ic_cc = (
        ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
          .filterDate(range_date[0], range_date[1])
          .filterBounds(ee.Geometry.Point(lon, lat))
    )
    ic_cc_pd = ic_cc.getRegion(
        geometry=ee.Geometry.Point(lon, lat),
        scale=edge_size
    ).getInfo()
    ic_cc_pd = pd.DataFrame(ic_cc_pd[1:], columns=ic_cc_pd[0])
    ic_cc_pd["img_id"] = ic_cc_pd["id"].apply(lambda x: "COPERNICUS/S2_HARMONIZED" + "/" + x)
    ic_cc_pd = ic_cc_pd.loc[:, ["img_id", "cs", "cs_cdf"]]
    
    # Join the tables  remove the right_on column
    table = table.merge(
        ic_cc_pd, left_on="img_id", right_on="img_id", how="right"
    )

    # Add the MGRS title
    table["mgrs_title"] = table["img_id"].apply(
        lambda x: pathlib.Path(x).stem.split("_")[2]
    )

    return table


def intermediate_process(
    table: pd.DataFrame,
    out_folder: str,
    process_function: Callable,
    process_function_args: dict,
    force: bool = False,
) -> pd.DataFrame:
    """Apply a process to a folder of images.

    Args:
        table (pd.DataFrame): The table with the images to process.
        out_folder (str): The output folder to save the results.
        process_function (Callable): The function to apply to the images.
        process_function_args (dict): The arguments to pass to the function.
        force (bool, optional): If True, the process is done again.
            Defaults to False.

    Raises:
        FileNotFoundError: If the input file does not exist.

    Returns:
        pathlib.Path: The output folder.
    """
    # Reset the table index
    table = table.copy()
    table.reset_index(drop=True, inplace=True)

    # Create the output csv
    out_table_name = (out_folder.parent) / (out_folder.name + ".csv")

    # Check if the output file exists
    if (not out_folder.exists()) or force:

        # Create the output folder if it does not exist
        out_folder.mkdir(parents=True, exist_ok=True)

        # Apply the cloud removal for each image
        out_table = process_function(
            table=table, out_folder=out_folder, **process_function_args
        )
        out_table.to_csv(out_table_name, index=False)
    else:
        out_table = pd.read_csv(out_table_name)

    return out_table


def cloudmasking_s2(
    table: pd.DataFrame,
    out_folder: pathlib.Path,
    sensor: Sentinel2,
    device: Union[str, torch.device],
    quiet: bool = False,
) -> pd.DataFrame:
    """Generate a cloud mask for a Sentinel-2 dataset.

    Args:
        in_folder (pathlib.Path): The input folder.
        out_folder (pathlib.Path): The output folder.
        spectral_embedding_weights_path (str): The path to the
            spectral embedding weights.
        cloudmask_weights_path (str): The path to the cloud mask weights.
        device (Union[str, torch.device]): The device to use.
        quiet (bool, optional): If True, the function does not print. Defaults
            to False.

    Returns:
        pd.DataFrame: The table with the cloud cover.
    """

    # Define Sentinel-2 descriptor
    s2_descriptor = [
        {
            "name": "B01",
            "band_type": "TOA Reflectance",
            "min_wavelength": 425.0,
            "max_wavelength": 461.0,
        },
        {
            "name": "B02",
            "band_type": "TOA Reflectance",
            "min_wavelength": 446.0,
            "max_wavelength": 542.0,
        },
        {
            "name": "B03",
            "band_type": "TOA Reflectance",
            "min_wavelength": 537.5,
            "max_wavelength": 582.5,
        },
        {
            "name": "B04",
            "band_type": "TOA Reflectance",
            "min_wavelength": 645.5,
            "max_wavelength": 684.5,
        },
        {
            "name": "B05",
            "band_type": "TOA Reflectance",
            "min_wavelength": 694.0,
            "max_wavelength": 714.0,
        },
        {
            "name": "B06",
            "band_type": "TOA Reflectance",
            "min_wavelength": 731.0,
            "max_wavelength": 749.0,
        },
        {
            "name": "B07",
            "band_type": "TOA Reflectance",
            "min_wavelength": 767.0,
            "max_wavelength": 795.0,
        },
        {
            "name": "B08",
            "band_type": "TOA Reflectance",
            "min_wavelength": 763.5,
            "max_wavelength": 904.5,
        },
        {
            "name": "B8A",
            "band_type": "TOA Reflectance",
            "min_wavelength": 847.5,
            "max_wavelength": 880.5,
        },
        {
            "name": "B09",
            "band_type": "TOA Reflectance",
            "min_wavelength": 930.5,
            "max_wavelength": 957.5,
        },
        {
            "name": "B10",
            "band_type": "TOA Reflectance",
            "min_wavelength": 1337.0,
            "max_wavelength": 1413.0,
        },
        {
            "name": "B11",
            "band_type": "TOA Reflectance",
            "min_wavelength": 1541.0,
            "max_wavelength": 1683.0,
        },
        {
            "name": "B12",
            "band_type": "TOA Reflectance",
            "min_wavelength": 2074.0,
            "max_wavelength": 2314.0,
        },
    ]
    cloudsen12_style = {0: 0, 1: 0, 2: 0, 3: 0, 6: 0, 4: 1, 3: 2, 5: 3}
    map_values = lambda x: cloudsen12_style.get(x, x)

    # Load the weights of the embedding model
    embedding_weights_path = sensor.weight_path / sensor.embedding_universal
    embedding_weights = torch.load(embedding_weights_path)

    # Load the weights of the cloud mask model 01
    cloudmask01_weights_path = sensor.weight_path / sensor.cloud_model_universal
    cloudmask01_weights = torch.load(cloudmask01_weights_path)

    # Load the weights of the cloud mask model 02
    cloudmask02_weights_path = sensor.weight_path / sensor.cloud_model_specific
    cloudmask02_weights = torch.load(cloudmask02_weights_path)

    # Model to device
    segmodel01 = phicloudmask.CloudMask(descriptor=s2_descriptor, device=device)
    segmodel01.embedding_model.load_state_dict(embedding_weights)
    segmodel01.cloud_model.load_state_dict(cloudmask01_weights)
    segmodel01.eval()
    segmodel01.to(device)

    # Auxiliar model
    segmodel02 = smp.Unet(
        encoder_name="mobilenet_v2", encoder_weights=None, in_channels=13, classes=4
    )
    segmodel02.load_state_dict(cloudmask02_weights)
    segmodel02.eval()
    segmodel02.to(device)

    # Iterate over all the images
    all_raw_files = [
        path / name for path, name in zip(table["folder"], table["outname"])
    ]
    new_cloud_covers = []
    for idx, file in enumerate(all_raw_files):
        if not quiet:
            print(f"Processing {file.name} [{idx + 1}/{len(all_raw_files)}]")

        # Read the image
        with rio.open(file) as src:
            s2_raw = src.read()
            metadata = src.profile
            metadata["nodata"] = 65535
            s2_raw_torch = torch.from_numpy(s2_raw[None] / 10000).float().to(device)

            # Obtain the cloud mask
            with torch.no_grad():
                # Create the mask for the first model
                cloud_probs_all = segmodel01(s2_raw_torch)
                cloud_mask_all = cloud_probs_all.argmax(dim=0).cpu().numpy()
                cloud_4class_all_01 = np.vectorize(map_values)(cloud_mask_all)

                # Create the mask for the second model
                cloud_probs_all_02 = segmodel02(s2_raw_torch).squeeze()
                cloud_mask_all_02 = cloud_probs_all_02.argmax(dim=0).cpu().numpy()
                cloud_4class_all_02 = cloud_mask_all_02

                # Combine the two masks
                cloud_4class_all = cloud_4class_all_01 + cloud_4class_all_02

            # Apply the cloud mask
            s2_clean = (s2_raw + 1) * (cloud_4class_all == 0)
            s2_clean[s2_clean == 0] = 65535
            s2_clean = s2_clean - 1
            s2_clean[s2_clean == 65534] = 65535

            # If more than 3 bands have zero values, then remove from all the bands
            outmask = (s2_clean == 0).sum(0) > 3
            s2_clean[:, outmask] = 65535

            # Remove 60 meters bands
            s2_clean = s2_clean[[1, 2, 3, 4, 5, 6, 7, 8, 11, 12]]

            # Get the cloud cover between 0 and 100
            cc_perc = (cloud_4class_all > 0).sum() / (
                cloud_4class_all.shape[0] * cloud_4class_all.shape[1]
            )

            # Save the cloud cover
            new_cloud_covers.append(cc_perc)

            # Save the image
            metadata["count"] = s2_clean.shape[0]
            with rio.open(out_folder / file.name, "w", **metadata) as dst:
                dst.write(s2_clean.astype(rio.uint16))

    # Update the cloud cover
    table["cloud_cover"] = new_cloud_covers

    return table


def gapfilling_s2(
    table: pd.DataFrame, out_folder: pathlib.Path, method: str, quiet: bool
) -> pd.DataFrame:
    """Remove gaps from a Sentinel-2 dataset.

    Args:
        table (pd.DataFrame): The table with the images to process.
        out_folder (pathlib.Path): The output folder.
        method (str): The method to fill the gaps.

    Returns:
        pathlib.Path: The path to the gap filled images.
    """

    # Load the data to fill the gaps
    all_raw_files = [
        path / name for path, name in zip(table["folder"], table["outname"])
    ]
    all_raw_dates = pd.to_datetime(table["img_date"])

    match_error = []
    for index, s2_img in enumerate(all_raw_files):

        # Load the s2 image and mask
        with rio.open(s2_img) as src:
            s2_data = src.read() / 10000
            s2_metadata = src.profile
            s2_data[s2_data == 6.5535] = np.nan
            s2_cloudmask = np.isnan(s2_data).mean(0)

        if s2_cloudmask.sum() == 0:
            # If there are no gaps, then just copy the image
            if not quiet:
                print(f"Processing {s2_img.name} [{index + 1}/{len(all_raw_files)}]")
            shutil.copy(s2_img, out_folder / s2_img.name)
            match_error.append(0)
        else:
            # closest n images in order to get the reference
            idxs = np.argsort(np.abs(all_raw_dates - all_raw_dates[index]))

            # Find the most appropriate image to fill the gap
            TOTAL_TRIES = 5
            counter = 0
            for i in idxs:
                # Load the reference image and mask
                with rio.open(all_raw_files[i]) as src:
                    s2_data_ref = src.read() / 10000
                    s2_data_ref[s2_data_ref == 6.5535] = np.nan
                    s2_cloudmask_ref = np.isnan(s2_data_ref) * 1.0

                # The reference image should have no gap
                condition = np.sum((s2_cloudmask_ref + s2_cloudmask) == 2)
                if condition == 0:

                    # Fill the gap
                    # There is three images:
                    # the image with gap (image1): This is the image that we want to fill the gap
                    # the reference image (image2): This is the image that we will use to fill the gap
                    # the final image (image3): The final image with the gap filled
                    image1, image2, image3 = s2_data, s2_data_ref, s2_data_ref
                    image1 = image1.copy()
                    image2 = image2.copy()
                    image3 = image3.copy()

                    ## Create a mask with the gaps
                    full_mask = ((s2_cloudmask + s2_cloudmask_ref) > 0) * 1.0

                    # mask -> 1: data, nan: gap
                    full_mask2 = full_mask.copy()
                    full_mask2[full_mask2 == 1] = np.nan
                    full_mask2[full_mask2 == 0] = 1

                    ## Mask image1 and image2
                    image1_masked = image1 * full_mask2
                    image2_masked = image2 * full_mask2

                    ## Apply histogram matching
                    new_image3 = np.zeros_like(image3)
                    for i in range(image3.shape[0]):
                        if method == "histogram_matching":
                            new_image3[i] = tripple_histogram_matching(
                                image1=image1_masked[i],
                                image2=image2_masked[i],
                                image3=image3[i],
                            )
                        elif method == "linear":
                            new_image3[i] = linear_interpolation(
                                image1=image1_masked[i],
                                image2=image2_masked[i],
                                image3=image3[i],
                            )

                    # Estimate metric (normalized difference)
                    a = new_image3[[2, 1, 0]].mean(0)
                    b = image1[[2, 1, 0]].mean(0)
                    metric = np.nanmean(np.abs(a - b) / (a + b))

                    # Prepare the final image
                    new_image3[np.isnan(new_image3)] = 0
                    image1[np.isnan(image1)] = 0
                    final_image = image1 + new_image3 * full_mask
                    final_image[final_image < 0] = 0

                    if counter == 0:
                        best_image = final_image
                        best_metric = metric
                    else:
                        if metric < best_metric:
                            best_image = final_image
                            best_metric = metric
                else:
                    continue

                # Try to fill the gap with the best image in 5 tries
                counter += 1
                if counter == TOTAL_TRIES:
                    break

            # Compare the final_image with the image
            best_image = (best_image * 10000).astype(np.uint16)
            if not quiet:
                print(
                    f"Processing {s2_img.name} [{index + 1}/{len(all_raw_files)}] with error {best_metric}"
                )

            # Save the image
            with rio.open(out_folder / s2_img.name, "w", **s2_metadata) as dst:
                dst.write(best_image)

            # Save the match error
            match_error.append(best_metric)

    # Save the match error
    table["match_error"] = match_error

    return table


def aligned_s2(
    table: pd.DataFrame, out_folder: str, quiet: bool = False
) -> pd.DataFrame:
    """Align all the images in the data cube.

    Args:
        table (pd.DataFrame): The table with the images to align.
        out_folder (str): The output folder.

    Returns:
        pd.DataFrame: The table with the images aligned.
    """

    # Load the data to fill the gaps
    all_raw_files = [
        path / name for path, name in zip(table["folder"], table["outname"])
    ]

    # Create the reference image using the last 10 images
    reference_files = all_raw_files[-10:]
    for idx, file in enumerate(reference_files):
        with rio.open(file) as src:
            s2_mean = src.read() / 10000
            if idx == 0:
                s2_acc = np.zeros_like(s2_mean)
            s2_acc = s2_acc + s2_mean
    s2_mean = s2_acc / len(reference_files)

    # Iterate over all the images
    align_error = []
    for idx, file in enumerate(all_raw_files):

        if not quiet:
            print(f"Processing {file.name} [{idx + 1}/{len(all_raw_files)}]")

        # Load the s2 image and mask
        with rio.open(file) as src:
            s2_data = src.read() / 10000
            metadata = src.profile

        # Set the alignment model
        syncmodel = satalign.PCC(
            datacube=s2_data[None],  # T x C x H x W
            reference=s2_mean,  # C x H x W
            upsample_factor=200,
            channel="luminance",
            crop_center=s2_mean.shape[2] // 2,
        )

        # Run the alignment
        news2cube, warps = syncmodel.run()

        # Save the aligned image
        news2cube = news2cube * 10000
        with rio.open(out_folder / file.name, "w", **metadata) as dst:
            dst.write(news2cube[0].astype(rio.uint16))

        # Save the warps after alignment
        error = (warps[0][0, 2], warps[0][1, 2])
        error = np.sqrt(error[0] ** 2 + error[1] ** 2)
        align_error.append(error)

    # Add the alignment error
    table["align_error"] = align_error

    return table


def tripple_histogram_matching(
    image1: np.ndarray, image2: np.ndarray, image3: np.ndarray
) -> np.ndarray:
    """Apply histogram matching to image3 using image1 and image2 as reference images.

    Args:
        image1 (np.ndarray): The first reference image.
        image2 (np.ndarray): The second reference image.
        image3 (np.ndarray): The image to be matched.

    Returns:
        np.ndarray: The matched image.
    """

    # remove nan values
    image1_nonan = image1.flatten().copy()
    image1_nonan = image1_nonan[~np.isnan(image1_nonan)]

    image2_nonan = image2.flatten().copy()
    image2_nonan = image2_nonan[~np.isnan(image2_nonan)]

    image3_nonan = image3.flatten().copy()
    image3_nonan = image3_nonan[~np.isnan(image3_nonan)]

    # Calculate histograms
    hist1, bins = np.histogram(image1_nonan, 128, [0, 2])
    hist2, bins = np.histogram(image2_nonan, 128, [0, 2])
    hist3, bins = np.histogram(image3_nonan, 128, [0, 2])

    # Calculate the cumulative distribution function (CDF) of img1
    cdf1 = hist1.cumsum() / hist1.sum()

    # Calculate the CDF of img2
    cdf2 = hist2.cumsum() / hist2.sum()

    # Create a lookup table (LUT) to map the pixel values of img1 to img2
    lut = np.interp(cdf2, cdf1, bins[:-1])

    # Perform histogram matching
    img3_matched = np.interp(image3.ravel(), bins[:-1], lut).reshape(image3.shape)

    return img3_matched


def linear_interpolation(
    image1: np.ndarray, image2: np.ndarray, image3: np.ndarray
) -> np.ndarray:
    """Apply linear interpolation to image3 using image1 and image2 as
    reference images.

    Args:
        image1 (np.ndarray): The first reference image.
        image2 (np.ndarray): The second reference image.
        image3 (np.ndarray): The image to be matched.

    Returns:
        np.ndarray: The matched image.
    """

    # remove nan values
    image1_nonan = image1.flatten().copy()
    image1_nonan = image1_nonan[~np.isnan(image1_nonan)]

    image2_nonan = image2.flatten().copy()
    image2_nonan = image2_nonan[~np.isnan(image2_nonan)]

    # Calculate the slope and intercept
    linreg = LinearRegression()
    linreg.fit(image2_nonan[:, np.newaxis], image1_nonan[:, np.newaxis])
    slope = linreg.coef_[0]
    intercept = linreg.intercept_

    # Apply the linear interpolation
    image3_matched = slope * image3 + intercept

    return image3_matched


def display_images(
    table: pd.DataFrame,
    out_folder: pathlib.Path,
    bands: List[int],
    ratio: int,
):
    """Display a GIF from a dataset.

    Args:
        interp_file (dict): The dataset to display.
        ratio (int, optional): The ratio to use. Defaults to 3000.
    """

    # Load the data
    all_raw_files = [
        path / name for path, name in zip(table["folder"], table["outname"])
    ]
    all_raw_files.sort()
    
    # Create the GIF from combined_s
    for index, file in enumerate(all_raw_files):        
        with rio.open(file) as src:
            data = src.read() / ratio

        # normalize the data according to the min and max values
        data = np.clip(data, 0, 1)

        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        img = np.moveaxis(data[bands, :, :], 0, -1)
        ax.imshow(img)
        ax.axis("off")
        title = "ID: " + str(index) + " - " + file.name
        plt.title(title, fontsize=20)
        plt.savefig(out_folder / f"temp_{index:03d}.png")
        plt.close()
        plt.clf()

    return None


def load_evoland(
    weights: str,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[Any, Any]:
    import onnxruntime as ort
    # ONNX inference session options
    so = ort.SessionOptions()
    so.intra_op_num_threads = 10
    so.inter_op_num_threads = 10
    so.use_deterministic_compute = False

    # Execute on cpu only
    if device == "cpu":
        ep_list = ["CPUExecutionProvider"]
    elif device == "cuda":
        ep_list = ["CUDAExecutionProvider"]
    else:
        raise ValueError("Invalid device")

    ort_session = ort.InferenceSession(
        weights,
        sess_options=so,
        providers=ep_list,
    )
    ort_session.set_providers(ep_list)
    ro = ort.RunOptions()

    return [ort_session, ro]


def super_s2(
    table: pd.DataFrame,
    out_folder: str,
    device: str,
    sensor: Sentinel2,
    quiet: bool = False
):  
    # super resolution requires the onnxruntime installed
    spec = importlib.util.find_spec("onnxruntime")
    if spec is None:
        raise ImportError(
            "onnxruntime is not installed. Please install it to use super resolution tools."
        )
    
    # Define the output file
    ort_session, ro = load_evoland(
        sensor.weight_path / sensor.super_model_specific,
        device=device,
    )

    # Load the data
    all_raw_files = [
        path / name for path, name in zip(table["folder"], table["outname"])
    ]

    # Iterate over all the images
    for idx, file in enumerate(all_raw_files):
        if not quiet:
            print(f"Processing {file.name} [{idx + 1}/{len(all_raw_files)}]")

        # Read the image
        with rio.open(file) as src:
            data = src.read()
            metadata = src.profile

        # Apply the super resolution
        sr = (
            ort_session.run(
                None, {"input": data[None].astype(np.float32)}, run_options=ro
            )[0]
            .squeeze()
            .astype(np.float16)
        )

        # Update the metadata
        metadata["width"] = sr.shape[1]
        metadata["height"] = sr.shape[2]
        metadata["transform"] = rio.Affine(
            metadata["transform"].a / 2,
            metadata["transform"].b,
            metadata["transform"].c,
            metadata["transform"].d,
            metadata["transform"].e / 2,
            metadata["transform"].f,
        )

        # Save the image
        with rio.open(out_folder / file.name, "w", **metadata) as dst:
            dst.write(sr.astype(rio.uint16))

    return table


def monthly_composites_s2(
    table: pd.DataFrame,
    out_folder: pathlib.Path,
    date_range: Tuple[str, str],
    agg_method: str,
    quiet: bool = False
):

    # Define the folder path using pathlib
    all_raw_files = [
        path / name for path, name in zip(table["folder"], table["outname"])
    ]

    # Load the first image to get the metadata
    with rio.open(all_raw_files[0]) as src:
        metadata = src.profile

    # Prepare the metadata
    all_raw_dates = pd.to_datetime(table["img_date"])
    all_raw_date_min = pd.to_datetime(date_range[0])
    all_raw_date_max = pd.to_datetime(date_range[1])
    all_raw_dates_unique = pd.date_range(
        all_raw_date_min, all_raw_date_max, freq="MS"
    ) + pd.DateOffset(days=14)
    all_raw_dates_unique = all_raw_dates_unique.strftime("%Y-%m-15")

    # Aggregate the data considering the method and dates
    new_table = []
    for idx, date in enumerate(all_raw_dates_unique):
        if not quiet:
            print(f"Processing {date} [{idx + 1}/{len(all_raw_dates_unique)}]")

        # Get the images to aggregate
        idxs = all_raw_dates.dt.strftime("%Y-%m-15") == date
        images = [all_raw_files[i] for i in np.where(idxs)[0]]

        if len(images) == 0:
            data = np.ones((metadata["count"], metadata["height"], metadata["width"]))
            data = 65535 * data
            nodata = 1
        else:
            # Read the images
            container = []
            for image in images:
                with rio.open(image) as src:
                    data = src.read()
                    metadata = src.profile
                container.append(data)

            # Aggregate the data
            if agg_method == "mean":
                data = np.mean(container, axis=0)
            elif agg_method == "median":
                data = np.median(container, axis=0)
            elif agg_method == "max":
                data = np.max(container, axis=0)
            elif agg_method == "min":
                data = np.min(container, axis=0)
            else:
                raise ValueError("Invalid aggregation method")

            nodata = 0

        # Save the image
        with rio.open(out_folder / f"{date}.tif", "w", **metadata) as dst:
            dst.write(data.astype(rio.uint16))

        meta_dict = {
            "img_date": date,
            "folder": out_folder,
            "outname": f"{date}.tif",
            "nodata": nodata,
        }

        new_table.append(meta_dict)

    return pd.DataFrame(new_table)


def interpolate_s2(
    table: pd.DataFrame,
    out_folder: pathlib.Path,
    quiet: bool = False
) -> pd.DataFrame:
    """Interpolate the missing values in a dataset.

    Args:
        table (pd.DataFrame): The table with the images to interpolate
        out_folder (pathlib.Path): The output folder.
        smooth_w (int, optional): The window length for the savgol
            filter. Defaults to 3.
        smooth_p (int, optional): The polynomial order for the savgol
            filter. Defaults to 1.

    Returns:
        pd.DataFrame: The table with the images interpolated.
    """
    if not quiet:
        print("Interpolating the missing values started...")

    # Load the data
    all_raw_files = [
        path / name for path, name in zip(table["folder"], table["outname"])
    ]
    all_raw_files.sort()
    all_raw_dates = pd.to_datetime(table["img_date"])
    
    # Create a datacube    
    metadata = rio.open(all_raw_files[0]).profile
    data_np = np.array([rio.open(file).read() for file in all_raw_files]) / 10000
    data_np[data_np==6.5535] = np.nan
    data_np = xr.DataArray(
        data=data_np,
        dims=("time", "band", "y", "x"),
        coords={"time": all_raw_dates, "band": range(10)},
    )

    # Interpolate the missing values
    data_np = data_np.interpolate_na(dim="time", method="linear")
    
    # Save the images
    for idx, file in enumerate(all_raw_files):
        current_data = data_np[idx].values
        date = pd.to_datetime(table["img_date"].iloc[idx]).strftime("%Y-%m-%d")
        with rio.open(out_folder / f"{date}.tif", "w", **metadata) as dst:
            dst.write((current_data * 10000).astype(np.uint16))

    return table


def smooth_s2(
    table: pd.DataFrame,
    out_folder: pathlib.Path,
    smooth_w: int,
    smooth_p: int,
    device: Union[str, torch.device],
    quiet: bool
) -> pd.DataFrame:
    """Interpolate the missing values in a dataset.

    Args:
        table (pd.DataFrame): The table with the images to interpolate
        out_folder (pathlib.Path): The output folder.
        smooth_w (int, optional): The window length for the savgol
            filter. Defaults to 3.
        smooth_p (int, optional): The polynomial order for the savgol
            filter. Defaults to 1.

    Returns:
        pd.DataFrame: The table with the images interpolated.
    """

    if not quiet:
        print("Smoothing the values started...")

    # Load the data
    all_raw_files = [
        path / name for path, name in zip(table["folder"], table["outname"])
    ]
    out_files = [out_folder / file.name for file in all_raw_files]
    
    # Create a datacube    
    metadata = rio.open(all_raw_files[0]).profile
    data_np = (np.array([rio.open(file).read() for file in all_raw_files]) / 10000).astype(np.float32)

    # Create monthly composites
    data_month = pd.to_datetime(table["img_date"]).dt.month
    data_clim = []
    for month in range(1, 13):
        data_clim.append(data_np[data_month == month].mean(axis=0) / 10000)
    data_clim = np.array(data_clim)

    # Create the residuals
    for idx, month in enumerate(data_month):
        data_np[idx] = data_np[idx] - data_clim[month - 1]

    # Smooth the residuals
    data_np = torch.from_numpy(data_np).float().to(device)
    try:
        data_np = (
            gaussian_smooth(
                data_np, kernel_size=smooth_w, sigma=smooth_p
            ).cpu().numpy()
        )
    except Exception as e:
        print(e)
        data_np = data_np.cpu().numpy()

    # add the residuals to the climatology
    for idx, month in enumerate(data_month):
        data_np[idx] = data_np[idx] + data_clim[month - 1]

    for idx, file in enumerate(out_files):
        with rio.open(file, "w", **metadata) as dst:
                dst.write((data_np[idx] * 10000).astype(np.uint16))

    # Prepare the new table
    new_table = pd.DataFrame(
        {
            "img_date": table["img_date"],
            "outname": table["outname"],
        }
    )

    return new_table



def gaussian_kernel1d(kernel_size: int, sigma: float):
    """
    Returns a 1D Gaussian kernel.
    """
    # Create a tensor with evenly spaced values centered at 0
    x = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    # Calculate the Gaussian function
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    # Normalize the kernel to ensure the sum of all elements is 1
    kernel = kernel / kernel.sum()
    return kernel


def gaussian_smooth(tensor, kernel_size: int, sigma: float):
    """
    Apply Gaussian smoothing on the time dimension (first dimension) of the input tensor.

    Args:
    - tensor (torch.Tensor): Input tensor of shape (T, C, H, W) where T is the time dimension.
    - kernel_size (int): Size of the Gaussian kernel.
    - sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
    - smoothed_tensor (torch.Tensor): Smoothed tensor.
    """
    # Get the Gaussian kernel
    kernel = gaussian_kernel1d(kernel_size, sigma).to(tensor.device).view(1, 1, -1)

    # Prepare the tensor for convolution: (B, C, T) where B = C*H*W, C=1, T=102
    T, C, H, W = tensor.shape
    tensor = tensor.view(T, -1).permute(1, 0).unsqueeze(1)  # Shape: (C*H*W, 1, T)

    # Apply convolution
    padding = kernel_size // 2
    smoothed = F.conv1d(tensor, kernel, padding=padding, groups=1)

    # Reshape back to original shape
    smoothed = smoothed.squeeze(1).permute(1, 0).view(T, C, H, W)

    return smoothed


def monthly_calendar(df, year1, year2, data_char="X", no_data_char="."):

    # Initialize the matrix for the calendar
    df = df.copy()
    df["img_date"] = pd.to_datetime(df["img_date"])

    years = range(year1, year2 + 1)
    months = range(1, 13)

    # Create a dictionary to count the months with data
    year_month_data = {year: [no_data_char] * 12 for year in years}

    # Populate the dictionary with data
    for year in years:
        for month in months:
            if not df[
                (df["img_date"].dt.year == year) & (df["img_date"].dt.month == month)
            ].empty:
                year_month_data[year][month - 1] = data_char

    # Print the matrix month as numbers
    print("Y/M".center(10) + " ".join([str(x) for x in range(1, 13)]))
    for year in years:
        print(f"{year:<10}", end="")
        for month in year_month_data[year]:
            print(f" {month}", end="")
        print()


def download_weights(
    path: Union[str, pathlib.Path],
    quiet: bool = False,
) -> pathlib.Path:
    """This function downloads the weights for the models.

    Args:
        path (Union[str, pathlib.Path]): The path to save the weights.
        quiet (bool, optional): If True, the function will not print
            the progress. Defaults to False.

    Returns:
        pathlib.Path: The path to the weights.
    """
    
    if not quiet:
        print("Downloading the satcube weights...")


    URI = "https://github.com/JulioContrerasH/satcube/releases/download/weights-v1.0/"
    path = pathlib.Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Download the weights
    s2_cloud_model_specific = "s2_cloud_model_specific.pt"
    s2_cloud_model_universal = "s2_cloud_model_universal.pt"
    s2_embedding_model_universal = "s2_embedding_model_universal.pt"
    s2_super_model_specific = "s2_super_model_specific.pt"

    # Download the weights
    if not (path / s2_cloud_model_specific).exists():
        with requests.get(URI + s2_cloud_model_specific) as r:        
            with open(path / s2_cloud_model_specific, "wb") as f:
                f.write(r.content)

    if not (path / s2_cloud_model_universal).exists():
        with requests.get(URI + s2_cloud_model_universal, stream=True) as r:        
            with open(path / s2_cloud_model_universal, "wb") as f:
                f.write(r.content)

    if not (path / s2_embedding_model_universal).exists():
        with requests.get(URI + s2_embedding_model_universal, stream=True) as r:        
            with open(path / s2_embedding_model_universal, "wb") as f:
                f.write(r.content)

    if not (path / s2_super_model_specific).exists():
        with requests.get(URI + s2_super_model_specific, stream=True) as r:        
            with open(path / s2_super_model_specific, "wb") as f:
                f.write(r.content)

    return path
