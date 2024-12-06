from typing import List
from dataclasses import dataclass
import logging

import numpy as np
import torch
from transformers import SamModel, SamProcessor, pipeline
from PIL import Image



# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Grounding%20DINO/GroundingDINO_with_Segment_Anything.ipynb
@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]
    
@dataclass
class DetectionResult:
    score: float
    bbox: BoundingBox


class GroundedSamSegmenter:
    """
    ## GroundedSamSegmenter
    
    A segmentor that makes use of Grounded SAM [1]. 
    
    You can give it a prompt and other parameters to control how the segmentation happens. Right
    now, it does some filters after segmentation based on xyz and number of points. These are to
    help ensure the segmentations are the objects of interest. This class relies on the
    transformers package from huggingface and the pipeline can be run on the CPU or GPU.
    
    **References**

    [1] "Grounded SAM: Assembling Open-World Models for Diverse Visual Tasks" (Ren et al 2024) 
        https://arxiv.org/abs/2401.14159
    """
    def __init__(
        self,
        device: str = "cpu",
        *,
        min_depth: float = 0.2,
        max_depth: float = 5.0,
        quantile_max: float = 0.5,
        prompt: str = "small object",
        threshold: float = 0.1,
        detector_id: str = "IDEA-Research/grounding-dino-base",
        segmenter_id: str = "facebook/sam-vit-base",
    ) -> None:
        self.prompt = prompt
        self.object_detector = pipeline(
            model=detector_id, task="zero-shot-object-detection", device=device
        )
        self.segment_anything = SamModel.from_pretrained(segmenter_id).to(device)
        self.processor = SamProcessor.from_pretrained(segmenter_id)
        self.threshold = threshold
        self.device = device
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.quantile_max = quantile_max

    def detect(self, image: Image.Image) -> List[DetectionResult]:
        """given a PIL Image, gets the detection results"""
        with torch.no_grad():
            results_dicts = self.object_detector(
                image, candidate_labels=[self.prompt], threshold=self.threshold
            )
        detection_results = [DetectionResult(
            score=result_dict["score"],
            bbox=BoundingBox(
                xmin=result_dict['box']['xmin'],
                ymin=result_dict['box']['ymin'],
                xmax=result_dict['box']['xmax'],
                ymax=result_dict['box']['ymax'],
            )
        ) for result_dict in results_dicts]
        return detection_results

    def segment(self, rgb: np.ndarray, xyz: np.ndarray) -> torch.Tensor:
        """Segments an RBG image and applies filters based on XYZ image"""
        # Convert data
        H, W, _ = rgb.shape
        image = Image.fromarray((rgb * 255).astype(np.uint8))
        rgb = torch.from_numpy(rgb).to(self.device)
        xyz = torch.from_numpy(xyz).to(self.device)

        # Grounded SAM segment (Grounding DINO + SAM)
        detections = self.detect(image)
        logging.debug(f"{len(detections)} detections found")
        boxes = [detection.bbox.xyxy for detection in detections]
        inputs = self.processor(
            images=image, input_boxes=[boxes], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():  # reduce memory
            outputs = self.segment_anything(**inputs)
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes
        )[0]  # (N, 3, H, W)
        masks = (torch.sum(masks, dim=1) > 0).to(torch.int64)  # (N, H, W) - reduce along dim of 3
        masks = get_id_masks(masks)
        logging.debug(f"masks unique: {torch.unique(masks)}; masks shape: {masks.shape}")
        masks = filter_masks(
            masks, 
            xyz, 
            quantile_max=self.quantile_max, 
            min_points=1000, 
            min_depth=self.min_depth, 
            max_depth=self.max_depth
        )
        # Return
        return masks


def get_id_masks(masks: torch.Tensor) -> torch.Tensor:
    """turns a (N, H, W) binary mask into a (H, W) int64 mask"""
    masks_background = (torch.sum(masks, dim=0) == 0).to(torch.int64).unsqueeze(0)
    masks_appended = torch.concat([masks_background, masks])
    labels = torch.argmax(masks_appended, dim=0)
    return labels

def filter_masks(
    masks: torch.Tensor,
    xyz: torch.Tensor,
    quantile_max: float,
    min_points: int,
    min_depth: float, 
    max_depth: float,
) -> torch.Tensor:
    """filters the masks according to depth, count, etc.
    
    Here are the filters that are run:

    1. Remove NaNs
    2. At least `min_points` points in the segment
    3. Median depth of each segment has to be within `min_depth` and `max_depth`
    4. XYZ quantiles outside of `quantile_max` are discarded
    5. Filter segment so each point is within `quantile_max` of median
    """
    masks[torch.sum(torch.isnan(xyz), dim=-1) > 0] = 0  # remove nans
    new_masks = torch.zeros_like(masks)
    depths = torch.norm(xyz, dim=-1)
    min_max_depth_mask = torch.logical_and(min_depth <= depths, depths <= max_depth)
    curr_new_id = 1
    for id in torch.unique(masks):
        if id == 0:
            continue  # ground plane
        masked_xyz = xyz[masks == id]
        depth_i = torch.norm(masked_xyz, dim=1)
        median = torch.median(depth_i)
        if median > max_depth or median < min_depth:
            logging.debug(f"depth min/max violation with id {id}")
            continue
        quantiles = torch.quantile(
            masked_xyz, 
            torch.tensor([0.25, 0.75], dtype=xyz.dtype, device=xyz.device), 
            dim=0
        )
        if torch.max(quantiles[1] - quantiles[0]) > quantile_max:
            logging.debug(f"scale violation with id {id}")
            continue
        med_centroid = torch.median(masked_xyz, dim=0).values  # (3,)
        dist_from_center = torch.norm(xyz - med_centroid, dim=-1)
        mask_nearby = dist_from_center <= quantile_max
        mask_for_id_new = torch.logical_and(mask_nearby, masks == id)
        mask_for_id_new = torch.logical_and(mask_for_id_new, min_max_depth_mask)
        logging.info(f"points: {torch.sum(mask_for_id_new)}")
        if torch.sum(mask_for_id_new) < min_points:
            logging.debug(f"min point violation with id {id} only had {torch.sum(mask_for_id_new)} points")
            continue  # not enough points of object
        new_masks[mask_for_id_new] = curr_new_id
        curr_new_id += 1
    return new_masks