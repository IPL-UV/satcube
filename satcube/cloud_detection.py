import torch

class LandsatCloudDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Define bit flags for clouds based on the
        # Landsat QA band documentation
        cloud_flags = (1 << 3) | (1 << 4) | (1 << 1)

        ## Get the QA band
        qa_band = x[6]
        mask_band = x[:6].mean(axis=0)
        mask_band[~torch.isnan(mask_band)] = 1

        ## Create a cloud mask
        cloud_mask = torch.bitwise_and(qa_band.int(), cloud_flags) == 0
        cloud_mask = cloud_mask.float()
        cloud_mask[cloud_mask == 0] = torch.nan
        cloud_mask[cloud_mask == 0] = 1
        final_mask = cloud_mask * mask_band
        return final_mask

