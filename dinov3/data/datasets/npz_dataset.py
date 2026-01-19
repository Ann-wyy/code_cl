# Custom dataset for loading images from NPZ files with CSV index
# Supports both single-channel and multi-channel images

import csv
import logging
import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from .decoders import Decoder, TargetDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")


class NpzImageDecoder(Decoder):
    """
    Decoder for NPZ image data.
    Handles both single-channel and multi-channel numpy arrays.
    """

    def __init__(self, image_array: np.ndarray, mode: str = "L") -> None:
        """
        Args:
            image_array: Numpy array of shape (H, W) or (H, W, C)
            mode: PIL image mode ("L" for grayscale, "RGB" for color)
        """
        self._image_array = image_array
        self._mode = mode

    def decode(self) -> Image.Image:
        """Convert numpy array to PIL Image."""
        array = self._image_array

        # Handle different array shapes
        if array.ndim == 2:
            # Grayscale image (H, W)
            mode = "L"
        elif array.ndim == 3:
            if array.shape[2] == 1:
                # Single channel (H, W, 1) -> squeeze to (H, W)
                array = array.squeeze(-1)
                mode = "L"
            elif array.shape[2] == 3:
                # RGB image (H, W, 3)
                mode = "RGB"
            else:
                raise ValueError(f"Unsupported channel count: {array.shape[2]}")
        else:
            raise ValueError(f"Unsupported array shape: {array.shape}")

        # Normalize to 0-255 range if needed
        if array.dtype == np.float32 or array.dtype == np.float64:
            # Assume float images are in [0, 1] range
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        elif array.dtype != np.uint8:
            # Convert other types to uint8
            array = array.astype(np.uint8)

        # Create PIL Image
        image = Image.fromarray(array, mode=mode)

        # Convert to RGB if requested (for compatibility with augmentations)
        if self._mode == "RGB" and mode == "L":
            image = image.convert("RGB")

        return image


class NpzDataset(ExtendedVisionDataset):
    """
    Custom dataset for loading images from NPZ files.

    Dataset structure:
        - CSV file: Contains list of NPZ file paths (one per line or with header)
        - NPZ files: Each contains image data as numpy array

    CSV format options:
        Option 1 - Simple list (no header):
            /path/to/image1.npz
            /path/to/image2.npz
            /path/to/image3.npz

        Option 2 - With header and columns:
            path,label
            /path/to/image1.npz,0
            /path/to/image2.npz,1

    NPZ file format:
        Each NPZ file should contain the image array under a key (default: 'image')
        Example: np.savez('img.npz', image=array)

    Config usage:
        dataset_path: NpzDataset:root=/path/to/data,csv_file=train.csv,npz_key=image

    Args:
        root: Root directory containing NPZ files
        csv_file: Path to CSV file (relative to root or absolute)
        npz_key: Key to access image array in NPZ file (default: 'image')
        image_mode: PIL image mode for output ('L' for grayscale, 'RGB' for color)
        csv_column: CSV column name or index for file paths (default: 0)
        has_header: Whether CSV file has a header row (default: False)
        transforms: Optional transforms to apply
        transform: Optional image transform
        target_transform: Optional target transform
    """

    def __init__(
        self,
        *,
        root: str,
        csv_file: str,
        npz_key: str = "image",
        image_mode: str = "L",
        csv_column: int | str = 0,
        has_header: bool = False,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=NpzImageDecoder,
            target_decoder=TargetDecoder,
        )

        self.root = Path(root)
        self.npz_key = npz_key
        self.image_mode = image_mode
        self.csv_column = csv_column
        self.has_header = has_header

        # Load file list from CSV
        csv_path = Path(csv_file)
        if not csv_path.is_absolute():
            csv_path = self.root / csv_path

        logger.info(f'Loading dataset from CSV: "{csv_path}"')
        self.file_paths = self._load_csv(csv_path)
        logger.info(f"Loaded {len(self.file_paths)} samples")

    def _load_csv(self, csv_path: Path) -> list[Path]:
        """Load file paths from CSV file."""
        file_paths = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)

            # Skip header if present
            if self.has_header:
                next(reader)

            for row in reader:
                if not row:  # Skip empty rows
                    continue

                # Get file path from specified column
                if isinstance(self.csv_column, int):
                    file_path = row[self.csv_column]
                else:
                    # If csv_column is a string, assume first row was header
                    # This is a simplification - for complex cases use pandas
                    file_path = row[0]

                # Convert to Path and make absolute if needed
                file_path = Path(file_path.strip())
                if not file_path.is_absolute():
                    file_path = self.root / file_path

                file_paths.append(file_path)

        return file_paths

    def get_image_data(self, index: int) -> np.ndarray:
        """Load image array from NPZ file."""
        npz_path = self.file_paths[index]

        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        try:
            # Load NPZ file
            data = np.load(npz_path)

            # Get image array
            if self.npz_key not in data:
                available_keys = list(data.keys())
                raise KeyError(
                    f"Key '{self.npz_key}' not found in NPZ file. "
                    f"Available keys: {available_keys}"
                )

            image_array = data[self.npz_key]

            # Pass mode to decoder via a wrapper
            # Note: We return the array, and __getitem__ will create the decoder
            return image_array

        except Exception as e:
            raise RuntimeError(f"Failed to load NPZ file {npz_path}: {e}") from e

    def get_target(self, index: int) -> Any:
        """
        Get target for the sample.
        For self-supervised learning (DINOv3), we don't need labels.
        Return None or index.
        """
        return None  # No labels needed for self-supervised learning

    def __getitem__(self, index: int):
        """Get a sample from the dataset."""
        try:
            # Get image array
            image_array = self.get_image_data(index)

            # Create decoder with mode
            decoder = NpzImageDecoder(image_array, mode=self.image_mode)
            image = decoder.decode()

        except Exception as e:
            raise RuntimeError(f"Cannot read image for sample {index}") from e

        # Get target (None for self-supervised)
        target = self.get_target(index)
        target = self.target_decoder(target).decode()

        # Apply transforms
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_paths)
