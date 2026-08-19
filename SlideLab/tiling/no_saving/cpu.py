import os
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info
from typing import Dict, Tuple, List, Optional, Any, Union
import cv2
import openslide
from PIL import Image

def skip_none_collate(batch):
    """
    Collate function to drop None items (tiles discarded by QC/blur/stain filters).
    Returns None if all items in the batch are None.
    """
    valid_batch = [item for item in batch if item is not None and item[0] is not None]
    if len(valid_batch) == 0:
        return None, None
    imgs, coords = zip(*valid_batch)
    imgs_tensor = torch.stack(imgs, dim=0)
    coords_tensor = torch.stack(coords, dim=0) if torch.is_tensor(coords[0]) else torch.tensor(np.array(coords), dtype=torch.float32)
    return imgs_tensor, coords_tensor

class StreamingTileDataset(Dataset):
    """
    Worker-safe Dataset for high-performance on-the-fly tile streaming without saving to disk.
    Ensures OpenSlide objects are initialized per DataLoader worker process to prevent fork crashes.
    """
    def __init__(
        self,
        slide: Union[str, openslide.OpenSlide],
        coordinates: List[Tuple[int, int]],
        adjusted_size: Tuple[int, int],
        desired_size: int = 256,
        *,
        pipeline_steps: Optional[List[Any]] = None,
        transforms: Optional[Any] = None,
        level: int = 0
    ):
        if isinstance(slide, str):
            self.slide_path = slide
            self._slide = None
        elif hasattr(slide, '_filename'):
            self.slide_path = slide._filename
            self._slide = slide
        else:
            self.slide_path = getattr(slide, 'filename', None)
            self._slide = slide

        self.coordinates = coordinates
        self.adjusted_size = adjusted_size if isinstance(adjusted_size, tuple) else (adjusted_size, adjusted_size)
        self.desired_size = desired_size
        self.pipeline_steps = pipeline_steps or []
        self.transforms = transforms
        self.level = level
        self.vars_dict = {}
        self._worker_slide = None

    def _get_slide(self) -> openslide.OpenSlide:
        worker_info = get_worker_info()
        if worker_info is not None:
            # Multi-process DataLoader worker: each worker must have its own OpenSlide instance
            if self._worker_slide is None:
                self._worker_slide = openslide.OpenSlide(self.slide_path)
            return self._worker_slide
        else:
            # Main process or single-threaded
            if self._slide is None and self.slide_path is not None:
                self._slide = openslide.OpenSlide(self.slide_path)
            return self._slide

    def __len__(self) -> int:
        return len(self.coordinates)

    def _read_region_np(self, coord: Tuple[int, int]) -> np.ndarray:
        slide = self._get_slide()
        region = slide.read_region(coord, self.level, self.adjusted_size).convert('RGB')
        region_np = np.array(region, dtype=np.uint8)
        if self.adjusted_size[0] != self.desired_size or self.adjusted_size[1] != self.desired_size:
            region_np = cv2.resize(
                region_np,
                (self.desired_size, self.desired_size),
                interpolation=cv2.INTER_LINEAR
            )
        return region_np

    def _preprocess_tile(self, img_np: np.ndarray, coord: Tuple[int, int]) -> Optional[np.ndarray]:
        for step in self.pipeline_steps:
            img_np = step(img_np, self.vars_dict, coord)
            if img_np is None:
                return None
        return img_np

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        coord = self.coordinates[idx]
        coord_tensor = torch.tensor(coord, dtype=torch.float32)

        try:
            img_np = self._read_region_np(coord)
            processed_np = self._preprocess_tile(img_np, coord)
            if processed_np is None:
                return None, coord_tensor

            if self.transforms:
                # transforms typically expect [C, H, W] tensor or PIL Image
                img_tensor = torch.from_numpy(processed_np).permute(2, 0, 1)  # HWC -> CHW
                img_tensor = self.transforms(img_tensor)
            else:
                img_tensor = torch.from_numpy(processed_np)
            return img_tensor, coord_tensor
        except Exception as e:
            return None, coord_tensor

# Alias for backwards compatibility
CPUTileDataset = StreamingTileDataset

class GPUTileDataset(Dataset):
    """
    Dataset for GPU processing without saving tiles, yielding raw CHW uint8/float tensors for batch GPU processing.
    """
    def __init__(
        self,
        slide: Union[str, openslide.OpenSlide],
        coordinates: List[Tuple[int, int]],
        adjusted_size: Tuple[int, int],
        desired_size: int = 256,
        *,
        pipeline_steps: Optional[List[Any]] = None,
        transforms: Optional[Any] = None,
        device: str = "cuda",
        level: int = 0
    ):
        if isinstance(slide, str):
            self.slide_path = slide
            self._slide = None
        elif hasattr(slide, '_filename'):
            self.slide_path = slide._filename
            self._slide = slide
        else:
            self.slide_path = getattr(slide, 'filename', None)
            self._slide = slide

        self.coordinates = coordinates
        self.adjusted_size = adjusted_size if isinstance(adjusted_size, tuple) else (adjusted_size, adjusted_size)
        self.desired_size = desired_size
        self.pipeline_steps = pipeline_steps or []
        self.transforms = transforms
        self.device = device
        self.level = level
        self._worker_slide = None

    def _get_slide(self) -> openslide.OpenSlide:
        worker_info = get_worker_info()
        if worker_info is not None:
            if self._worker_slide is None:
                self._worker_slide = openslide.OpenSlide(self.slide_path)
            return self._worker_slide
        else:
            if self._slide is None and self.slide_path is not None:
                self._slide = openslide.OpenSlide(self.slide_path)
            return self._slide

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        coord = self.coordinates[idx]
        coord_tensor = torch.tensor(coord, dtype=torch.float32)
        slide = self._get_slide()
        region = slide.read_region(coord, self.level, self.adjusted_size).convert('RGB')
        region_np = np.array(region, dtype=np.uint8)
        if self.adjusted_size[0] != self.desired_size or self.adjusted_size[1] != self.desired_size:
            region_np = cv2.resize(
                region_np,
                (self.desired_size, self.desired_size),
                interpolation=cv2.INTER_LINEAR
            )
        img_tensor = torch.from_numpy(region_np) # HWC uint8
        return img_tensor, coord_tensor

