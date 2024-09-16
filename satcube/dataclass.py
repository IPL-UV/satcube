import pathlib
from datetime import datetime
from typing import List, Optional

import pydantic


class Sensor(pydantic.BaseModel):
    start_date: str
    end_date: str
    edge_size: int
    bands: List[str]


class Sentinel2(Sensor):
    weight_path: pathlib.Path
    start_date: Optional[str] = "2015-06-27"
    end_date: Optional[str] = datetime.now().strftime("%Y-%m-%d")
    resolution: Optional[int] = 10
    edge_size: Optional[int] = 384
    embedding_universal: Optional[str] = "s2_embedding_model_universal.pt"
    cloud_model_universal: str = "s2_cloud_model_universal.pt"
    cloud_model_specific: str = "s2_cloud_model_specific.pt"
    super_model_specific: str = "s2_super_model_specific.pt"
    bands: List[str] = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B10",
        "B11",
        "B12",
    ]
