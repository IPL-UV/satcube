import pathlib
from typing import Optional, Tuple, Union
from datetime import datetime

import fastcubo
import pandas as pd
import torch

from satcube.dataclass import Sensor
from satcube.utils import (aligned_s2, cloudmasking_s2, display_images,
                           gapfilling_s2, intermediate_process, interpolate_s2,
                           metadata_s2, monthly_composites_s2, smooth_s2, super_s2)


class SatCube:
    """Satellite cube class to create datacubes from a specific sensor."""

    def __init__(
        self,
        sensor: Sensor,
        output_dir: str,
        max_workers: int,
        coordinates: Tuple[float, float],
        device: Union[str, torch.device],
    ):
        """Create a new instance of the Satellite Cube class.

        Args:
            coordinates (Tuple[float, float]): The coordinates of the
                location to download the data.
            sensor (Sensor): The sensor object with all the information
                to download and preprocess the data.
            output_dir (str): The output directory to save the data.
            max_workers (int): The maximum number of workers to use in the
                download process.
            device (Union[str, torch.device]): The device to use in the
                cloud removal process.
        """
        self.device = device
        self.sensor = sensor
        self.output_dir = pathlib.Path(output_dir)
        self.max_workers = max_workers
        self.lon, self.lat = coordinates

        # If the output directory does not exist, create it
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def metadata_s2(
        self,
        out_csv: Optional[pathlib.Path] = None,
        quiet: Optional[pathlib.Path] = False,
        force: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Create a pd.DataFrame with the images to download and their
        cloud cover percentage. The table is saved in a CSV file.

        Args:
            force (Optional[bool], optional): If True, the query
                process is done again. Defaults to False.
            out_csv (Optional[pathlib.Path], optional): The path to the
                CSV file with the query table. Defaults to None.
            quiet (Optional[bool], optional): If True, no message is
                displayed. Defaults to False.
            force (Optional[bool], optional): If True, the query process
                is done again. Defaults to False.

        Returns:
            pd.DataFrame: The table with the images to download.
        """

        if out_csv is None:
            out_csv: pathlib.Path = self.output_dir / "s2_01_gee_query.csv"
        
        if not out_csv.exists() or force:
            query_table: pd.DataFrame = metadata_s2(
                lon=self.lon,
                lat=self.lat,
                range_date=(self.sensor.start_date, self.sensor.end_date),
                edge_size=self.sensor.edge_size,
                quiet=quiet
            )
            query_table.to_csv(out_csv, index=False)
        else:
            query_table = pd.read_csv(out_csv)

        return query_table

    def download_s2_image(
        self,
        table: pd.DataFrame,
        out_folder: Optional[pathlib.Path] = None,
        quiet: Optional[bool] = False,
        force: Optional[bool] = False,
    ) -> pathlib.Path:
        """Download the images from the query table.

        Args:
            table (pd.DataFrame): The table with the images to download.
            out_csv (Optional[pathlib.Path], optional): The path to the
                CSV file with the query table. Defaults to None.
            quiet (Optional[bool], optional): If True, the download
                process is not displayed. Defaults to False.
            force (Optional[bool], optional): If True, the download
                process is done again. Defaults to False.

        Returns:
            pathlib.Path: The path to the folder with the downloaded images.
        """

        # Create the output directory if it does not exist
        if out_folder is None:
            output_path: pathlib.Path = self.output_dir / "s2_01_raw"

        # Download the selected images
        if not output_path.exists() or force:
            if not quiet:
                print(f"Saving the images in the directory {output_path}")

            fastcubo.getPixels(
                table=table, nworkers=self.max_workers, output_path=output_path
            )

        # Add folder path
        table["folder"] = output_path

        return table

    def cloudmasking_s2(
        self,
        table: pd.DataFrame,
        out_folder: Optional[pathlib.Path] = None,
        quiet: Optional[bool] = False,
        force: Optional[bool] = False,
    ) -> pathlib.Path:
        """Remove the clouds from the data.

        Args:
            table (pd.DataFrame): The table with the images to remove
                the clouds.
            out_csv (Optional[pathlib.Path], optional): The path to the
                CSV file with the query table. Defaults to None.
            quiet (Optional[bool], optional): If True, the messages
                are not displayed. Defaults to False.
            force (Optional[bool], optional): If True, the cloud removal
                is done again. Defaults to False.

        Returns:
            pathlib.Path: The path to the folder with the
                data without clouds.
        """

        if out_folder is None:
            out_folder: pathlib = self.output_dir / "s2_02_nocloud"

        # Apply the cloud removal
        out_table = intermediate_process(
            table=table,
            out_folder=out_folder,
            process_function=cloudmasking_s2,
            process_function_args=dict(
                device=self.device,
                sensor=self.sensor,
                quiet=quiet
            ),
            force=force,
        )

        # Change the folder path
        out_table["folder"] = out_folder

        # Sort by cloud cover
        out_table = out_table.sort_values(by="cloud_cover", ascending=False)
        out_table.reset_index(drop=True, inplace=True)

        return out_table

    def gapfilling_s2(
        self,
        table: pd.DataFrame,
        method: Optional[str] = "linear",
        out_folder: Optional[pathlib.Path] = None,
        quiet: Optional[bool] = False,
        force: Optional[bool] = False,
    ) -> pathlib.Path:
        """Fill the gaps in the data.

        Args:
            force (Optional[bool], optional): If True, the gap filling
                is done again. Defaults to False.
            histogram_match_error (float, optional): If the error in the
                histogram matching is greater than this value, the image
                is not filled and therefore, it is removed. Defaults
                to 0.10.

        Returns:
            pathlib.Path: The path to the folder with the
                data without gaps.
        """

        if out_folder is None:
            out_folder: pathlib = self.output_dir / "s2_03_nogaps"

        # Apply the cloud removal
        out_table = intermediate_process(
            table=table,
            out_folder=out_folder,
            process_function=gapfilling_s2,
            process_function_args={"method": method, "quiet": quiet},
            force=force,
        )

        # Change the folder path
        out_table["folder"] = out_folder

        # Sort by the matching error
        out_table = out_table.sort_values(by="match_error", ascending=False)

        return out_table

    def align_s2(
        self,
        table: pd.DataFrame,
        out_folder: Optional[pathlib.Path] = None,
        quiet: Optional[bool] = False,
        force: Optional[bool] = False,
    ) -> pathlib.Path:
        """Align all the images in the data cube.

        Args:
            table (pd.DataFrame): The table with the images to align.
            force (Optional[bool], optional): If True, the alignment
                is done again. Defaults to False.

        Returns:
            pathlib.Path: The path to the folder with the
                aligned images.
        """

        if out_folder is None:
            out_folder: pathlib = self.output_dir / "s2_04_aligned"

        # Apply the cloud removal
        out_table = intermediate_process(
            table=table,
            out_folder=out_folder,
            process_function=aligned_s2,
            process_function_args={"quiet": quiet},
            force=force,
        )

        # Change the folder path
        out_table["folder"] = out_folder

        return out_table
    
    def monthly_composites_s2(
        self,
        table: Optional[pd.DataFrame],
        out_folder: Optional[pathlib.Path] = None,
        agg_method: Optional[str] = "median",
        date_range: Tuple[str, str] = ("2016-01-01", datetime.now().strftime("%Y-%m-%d")),
        quiet: Optional[bool] = False,
        force: Optional[bool] = False,
    ) -> pathlib.Path:
        """Smooth the data considering the temporal dimension.

        Args:
            force (Optional[bool], optional): If True, the interpolation
                is done again. Defaults to False.

        Returns:
            xr.Dataset: The interpolated data.
        """

        if out_folder is None:
            out_folder: pathlib = self.output_dir / "s2_05_monthlycomposites"

        # Prepare the composites
        out_table = intermediate_process(
            table=table,
            out_folder=out_folder,
            process_function=monthly_composites_s2,
            process_function_args=dict(
                agg_method=agg_method,
                date_range=date_range,
                quiet=quiet
            ),
            force=force
        )

        # Change the folder path
        out_table["folder"] = out_folder

        return out_table

    def interpolate_s2(
        self,
        table: pd.DataFrame,
        out_folder: Optional[pathlib.Path] = None,
        quiet: Optional[bool] = False,
        force: Optional[bool] = False,
    ) -> pathlib.Path:
        """Interpolate the data.

        Args:
            force (Optional[bool], optional): If True, the interpolation
                is done again. Defaults to False.

        Returns:
            xr.Dataset: The interpolated data.
        """

        if out_folder is None:
            out_folder: pathlib = self.output_dir / "s2_06_interpolation"
        
        # Apply the cloud removal
        out_table = intermediate_process(
            table=table,
            out_folder=out_folder,
            process_function=interpolate_s2,
            process_function_args=dict(
                quiet=quiet
            ),
            force=force,
        )

        # Change the folder path
        out_table["folder"] = out_folder

        return out_table
    
    def smooth_s2(
        self,
        table: pd.DataFrame,
        out_folder: Optional[pathlib.Path] = None,
        smooth_w: Optional[int] = 7,
        smooth_p: Optional[int] = 1,
        device: Union[str, torch.device, None] = None,
        quiet: Optional[bool] = False,
        force: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Interpolate the data.

        Args:
            force (Optional[bool], optional): If True, the interpolation
                is done again. Defaults to False.

        Returns:
            xr.Dataset: The interpolated data.
        """        
        if out_folder is None:
            out_folder: pathlib = self.output_dir / "s2_07_smoothed"

        # Apply the cloud removal
        out_table = intermediate_process(
            table=table,
            out_folder=out_folder,
            process_function=smooth_s2,
            process_function_args=dict(
                quiet=quiet,
                smooth_w=smooth_w,
                smooth_p=smooth_p,
                device=device if device is not None else self.device
            ),
            force=force,
        )

        # Change the folder path
        out_table["folder"] = out_folder

        return out_table


    def super_s2(
        self,
        table: pd.DataFrame,
        out_folder: Optional[pathlib.Path] = None,
        quiet: Optional[bool] = False,
        force: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Superresolution to the Sentinel-2 image cube.

        Args:
            table (pd.DataFrame): The table with the images to
                superresolve.
            out_folder (Optional[pathlib.Path], optional): The path to the
                CSV file with the query table. Defaults to None.
            quiet (Optional[bool], optional): If True, the messages
                are not displayed. Defaults to False.
            force (Optional[bool], optional): If True, the superresolution
                is done again. Defaults to False.

        Returns:
            pd.DataFrame: The table with the superresolved images.
        """

        if out_folder is None:
            out_folder: pathlib.Path = self.output_dir / "s2_08_superresolution"

        # Apply the Superresolution process
        out_table = intermediate_process(
            table=table,
            out_folder=out_folder,
            process_function=super_s2,
            process_function_args=dict(
                device=self.device,
                sensor=self.sensor,
                quiet=quiet
            ),
            force=force,
        )

        # Change the folder path
        out_table["folder"] = out_folder

        return out_table

    def display_images(
        self,
        table: pd.DataFrame,
        out_folder: Optional[pathlib.Path] = None,
        bands: Optional[list[str]] = [2, 1, 0],
        ratio: Optional[int] = 3000,
    ) -> pathlib.Path:
        """ Display the images in the table.

        Args:
            table (pd.DataFrame): The table with the images to display.
            out_folder (Optional[pathlib.Path], optional): The path to the
                CSV file with the query table. Defaults to None.
            bands (Optional[list[str]], optional): The bands to display.
                Defaults to [2, 1, 0].
            ratio (Optional[int], optional): The ratio to divide the
                image. Defaults to 3000. The larger the number, the
                darker the image.

        Returns:
            pathlib.Path: The path to the folder with the
                displayed images.
        """

        # Create the output folder
        out_folder = (
            self.output_dir / ("z_" + table["folder"].iloc[0].name + "_png")
        )
        out_folder.mkdir(exist_ok=True, parents=True)
        
        return display_images(
            table=table,
            out_folder=out_folder,
            bands=bands,
            ratio=ratio,
        )
