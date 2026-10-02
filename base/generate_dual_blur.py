"""Precompute saliency- and center-guided blurred images."""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import cv2
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SaliencyMapGenerator:
    """Generate saliency maps with U2Net or OpenCV fallback methods."""
    def __init__(self, use_gpu=True, backend='auto'):
        self.backend = backend
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.session = None
        self.rembg_available = False

        if backend in ('fine-grained', 'spectral-residual'):
            if not hasattr(cv2, 'saliency'):
                raise RuntimeError('OpenCV saliency requires opencv-contrib-python. Install requirements.txt.')
            return

        try:
            from rembg import new_session, remove

            # Set the model cache directory
            model_dir = os.environ.get('U2NET_HOME', str(Path(__file__).resolve().parents[1] / 'temporary/models/u2net'))
            os.makedirs(model_dir, exist_ok=True)
            os.environ['U2NET_HOME'] = model_dir

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']

            self.session = new_session(
                model_name='u2net',
                providers=providers
            )

            # Test
            test_img = Image.new('RGB', (10, 10), color='red')
            remove(test_img, session=self.session)

            self.rembg_available = True
            logging.info(f"rembg initialized successfully [{'GPU' if self.use_gpu else 'CPU'} mode]")

        except Exception as e:
            if backend == 'u2net':
                raise RuntimeError('Cannot initialize the requested U2Net backend') from e
            logging.warning(f"rembg initialization failed: {e}\nUsing the fallback method")

    def generate_saliency_map(self, img_path, target_size=(224, 224)):
        """Return a normalized saliency map with shape [H, W]."""
        try:
            if self.rembg_available:
                return self._generate_with_rembg(img_path, target_size)
            else:
                return self._generate_fallback(img_path, target_size)
        except Exception as e:
            logging.error(f"Failed to generate saliency map {img_path}: {e}")
            if self.backend != 'auto':
                raise
            return self._generate_fallback(img_path, target_size)

    def _generate_with_rembg(self, img_path, target_size):
        """Extract the foreground alpha map with rembg."""
        import rembg
        input_image = Image.open(img_path).convert('RGB')

        # Extract the saliency map
        output = rembg.remove(input_image, session=self.session, only_mask=True)
        saliency_map = np.array(output).astype(np.float32)

        # Process on GPU
        if self.use_gpu:
            saliency_map = self._process_on_gpu(saliency_map)
        else:
            saliency_map = self._process_on_cpu(saliency_map)

        # Resize
        if saliency_map.shape[:2] != target_size:
            saliency_map = cv2.resize(saliency_map, target_size)

        return saliency_map

    def _generate_fallback(self, img_path, target_size):
        """Compute OpenCV saliency, respecting an explicitly selected backend."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path}")

        # Resize
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size)

        saliency_map = None


        try:
            if self.backend == 'spectral-residual':
                raise ValueError('Skip Fine-Grained for the selected backend')
            saliency = cv2.saliency.StaticSaliencyFineGrained_create()
            success, saliency_map = saliency.computeSaliency(img)
            if success:

                saliency_map = saliency_map.astype(np.float32)
                # Normalize values in [0, 255]
                if saliency_map.max() > 1.0:
                    saliency_map = saliency_map / 255.0
                logging.debug("Fine-Grained saliency detection succeeded")
            else:
                saliency_map = None
        except Exception as e:
            if self.backend == 'fine-grained':
                raise RuntimeError('Fine-Grained saliency detection failed') from e
            logging.debug(f"Fine Grained failed: {e}")


        if saliency_map is None:
            if self.backend == 'fine-grained':
                raise RuntimeError('Fine-Grained saliency detection returned no map')
            try:
                saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
                success, saliency_map = saliency.computeSaliency(img)
                if success:

                    saliency_map = saliency_map.astype(np.float32)
                    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
                    logging.debug("Spectral Residual saliency detection succeeded")
                else:
                    saliency_map = None
            except Exception as e:
                if self.backend == 'spectral-residual':
                    raise RuntimeError('Spectral Residual saliency detection failed') from e
                logging.debug(f"Spectral Residual failed: {e}")

        if saliency_map is None:
            raise RuntimeError('No saliency map was produced; check the OpenCV contrib installation')
        return saliency_map

    def _process_on_gpu(self, saliency_map):
        """Normalize the saliency map on GPU, with a CPU fallback."""
        try:
            tensor_map = torch.from_numpy(saliency_map).to(self.device)
            tensor_map = (tensor_map - tensor_map.min()) / (tensor_map.max() - tensor_map.min() + 1e-8)
            return tensor_map.cpu().numpy()
        except Exception as e:
            logging.warning(f"GPU processing failed; falling back to CPU: {e}")
            return self._process_on_cpu(saliency_map)

    def _process_on_cpu(self, saliency_map):
        """Normalize the saliency map on CPU."""
        return (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)


class SaliencyGuidedBlur:
    """Blend original and Gaussian-blurred pixels using normalized saliency."""
    def __init__(self, blur_kernel_size=51, max_sigma=None):
        """Initialize the blur kernel and strength parameters."""
        self.blur_kernel_size = blur_kernel_size
        self.max_sigma = max_sigma

    def apply(self, img_array, saliency_map):
        """Apply spatially weighted blur to a normalized RGB image [H, W, 3]."""

        saliency_max = saliency_map.max()

        if saliency_max > 1e-6:  # Avoid division by zero
            saliency_normalized = saliency_map / saliency_max  # [0, 1]
        else:
            saliency_normalized = np.zeros_like(saliency_map)

        # Convert to uint8 for OpenCV
        img_uint8 = (img_array * 255).astype(np.uint8)

        # Generate the Gaussian-blurred image
        blurred_uint8 = cv2.GaussianBlur(
            img_uint8,
            (self.blur_kernel_size, self.blur_kernel_size),
            self.max_sigma if self.max_sigma else 0
        )

        # Convert back to float32
        blurred_float = blurred_uint8.astype(np.float32) / 255.0

        # Blend using saliency weights
        # Maximum saliency (1): blur_weight=0.5
        # Minimum saliency (0): blur_weight=1.0
        blur_weight = 0.5 + 0.5 * (1 - saliency_normalized)
        blur_weight_3d = blur_weight[:, :, np.newaxis]

        output = (1 - blur_weight_3d) * img_array + blur_weight_3d * blurred_float

        return np.clip(output, 0, 1)


class CenterGuidedBlur:
    """Apply a radial foveal blur profile centered on the image."""
    def __init__(self, h=224, w=224, blur_kernel_size=51, curve_type='exp', system_g=3):
        self.h = h
        self.w = w
        self.blur_kernel_size = blur_kernel_size
        self.curve_type = curve_type
        self.system_g = system_g

        # Precompute radial weights
        self.mask = self._compute_fovea_mask()

    def _compute_fovea_mask(self):
        """Compute radial preservation weights with the configured decay function."""
        center_x, center_y = self.w // 2, self.h // 2
        y, x = np.ogrid[:self.h, :self.w]

        # Compute each pixel's distance to the center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        # Use the center-to-origin distance as the maximum
        max_distance = np.sqrt(center_x**2 + center_y**2)

        # Normalize radial distances
        norm_distance = np.clip(distance / max_distance, 0, 1)

        # Parameters for medium blur strength
        c = 0.5
        center_resolution = 1 - c  # 0.5
        edge_resolution = 0

        # Apply the decay function
        if self.curve_type == 'exp':
            y0 = np.exp(-self.system_g * norm_distance)
        elif self.curve_type == 'linear':
            y0 = 1 - norm_distance
        elif self.curve_type == 'quadratic':
            y0 = 1 - norm_distance ** 2
        elif self.curve_type == 'log':
            b = 1 / (np.e - 1)
            a = np.log(b) + 1
            y0 = a - np.log(norm_distance + b)
        else:
            y0 = np.exp(-self.system_g * norm_distance)  # Default to exponential decay

        # Compute the final mask
        mask = edge_resolution + (center_resolution - edge_resolution) * y0
        return mask.astype(np.float32)

    def alpha_blend(self, img1, img2, mask):
        """Blend two images with a spatial mask."""
        mask_3d = mask[:, :, np.newaxis]
        blended = img1 * (1 - mask_3d) + img2 * mask_3d
        return np.clip(blended, 0, 1)

    def apply(self, img_array):
        """Apply spatially weighted blur to a normalized RGB image [H, W, 3]."""
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Generate the Gaussian-blurred image
        blurred_bgr = cv2.GaussianBlur(
            img_bgr,
            (self.blur_kernel_size, self.blur_kernel_size),
            0
        )

        # Convert back to RGB
        blurred_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Apply foveal blending
        blended = self.alpha_blend(img_array, blurred_rgb, 1 - self.mask)

        return blended


def check_existing_files(output_dir, img_list):
    """Return images for which either dual-blur output is missing."""
    saliency_dir = os.path.join(output_dir, 'saliency_guided')
    center_dir = os.path.join(output_dir, 'center_guided')

    to_process = []

    for img_path in img_list:
        # Build the output path: split/class/image.npy
        base_name = os.path.splitext(img_path)[0]  # Remove the extension
        save_name = base_name + '.npy'

        # Full file path
        saliency_file = os.path.join(saliency_dir, save_name)
        center_file = os.path.join(center_dir, save_name)

        # Skip only when both output files exist
        if os.path.exists(saliency_file) and os.path.exists(center_file):
            continue

        to_process.append(img_path)

    skipped = len(img_list) - len(to_process)
    if skipped > 0:
        logging.info(f" Skipping {skipped} existing images")

    return to_process


def generate_dual_blur_images(
    data_dir='../data/things-meg',
    output_dir=None,
    use_gpu=True,
    saliency_config=None,
    center_config=None,
    force_regenerate=False,
    save_saliency_maps=True,
    saliency_backend='auto',
    limit=None
):
    """Generate paired float32 [3, 224, 224] arrays from Image_set_Resize.

    Outputs retain train_images/test_images and category subdirectories under
    saliency_guided and center_guided. Existing pairs are skipped unless forced."""
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'Dual_Blur_Images')

    # Set default blur configuration
    if saliency_config is None:
        saliency_config = {
            'blur_kernel_size': 51,
            'max_sigma': None
        }

    if center_config is None:
        center_config = {
            'blur_kernel_size': 51,
            'curve_type': 'exp',
            'system_g': 3
        }

    # Create output directories
    saliency_guided_dir = os.path.join(output_dir, 'saliency_guided')
    center_guided_dir = os.path.join(output_dir, 'center_guided')

    os.makedirs(saliency_guided_dir, exist_ok=True)
    os.makedirs(center_guided_dir, exist_ok=True)

    if save_saliency_maps:
        saliency_maps_dir = os.path.join(output_dir, 'saliency_maps')
        os.makedirs(saliency_maps_dir, exist_ok=True)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Collect images from train/test splits
    img_root = os.path.join(data_dir, 'Image_set_Resize')
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f'Image directory does not exist: {img_root}')
    img_list = []

    for split in ['train_images', 'test_images']:  # Process both splits
        split_path = os.path.join(img_root, split)
        if not os.path.isdir(split_path):
            logging.warning(f" Cannot find {split} directory; skipping")
            continue

        for class_dir in sorted(os.listdir(split_path)):
            class_path = os.path.join(split_path, class_dir)
            if not os.path.isdir(class_path):
                continue

            for img_name in sorted(os.listdir(class_path)):
                if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
                    # Relative path: split/class/image.jpg
                    img_list.append(os.path.join(split, class_dir, img_name))

    logging.info(f"Found {len(img_list)} images")
    if not img_list:
        raise FileNotFoundError(f'No stimulus images found under {img_root}')

    # Check existing output files
    if not force_regenerate:
        img_list = check_existing_files(output_dir, img_list)
        logging.info(f"Images to process: {len(img_list)} images")

    if len(img_list) == 0:
        logging.info("All images exist; no processing needed")
        return

    if limit is not None:
        img_list = img_list[:limit]
    saliency_map_gen = SaliencyMapGenerator(use_gpu=use_gpu, backend=saliency_backend)
    saliency_blur_gen = SaliencyGuidedBlur(**saliency_config)
    center_blur_gen = CenterGuidedBlur(h=224, w=224, **center_config)

    # Process images
    success_count = 0
    fail_count = 0

    for img_path in tqdm(img_list, desc=" Generating dual-blur images"):
        full_img_path = os.path.join(img_root, img_path)

        try:
            # 1. Load the original image
            img = Image.open(full_img_path).convert('RGB')
            img_tensor = transform(img)
            img_array = img_tensor.permute(1, 2, 0).numpy()

            # 2. Generate the saliency map
            saliency_map = saliency_map_gen.generate_saliency_map(full_img_path)

            # 3. Generate saliency-guided blur
            saliency_blur_array = saliency_blur_gen.apply(img_array, saliency_map)
            saliency_blur_tensor = torch.from_numpy(saliency_blur_array).permute(2, 0, 1).float()

            # 4. Generate center-guided blur
            center_blur_array = center_blur_gen.apply(img_array)
            center_blur_tensor = torch.from_numpy(center_blur_array).permute(2, 0, 1).float()

            # 5. Save with the split/category directory structure
            split_and_class = os.path.dirname(img_path)

            # Create directories
            os.makedirs(os.path.join(saliency_guided_dir, split_and_class), exist_ok=True)
            os.makedirs(os.path.join(center_guided_dir, split_and_class), exist_ok=True)

            # Save arrays
            save_name = os.path.splitext(img_path)[0] + '.npy'

            np.save(os.path.join(saliency_guided_dir, save_name), saliency_blur_tensor.numpy())
            np.save(os.path.join(center_guided_dir, save_name), center_blur_tensor.numpy())

            # Optionally save the saliency map
            if save_saliency_maps:
                os.makedirs(os.path.join(saliency_maps_dir, split_and_class), exist_ok=True)
                np.save(os.path.join(saliency_maps_dir, save_name), saliency_map)

            success_count += 1

        except Exception as e:
            logging.error(f"Processing failed {img_path}: {e}")
            fail_count += 1
            continue

    # Report generation statistics
    logging.info(f"\n{'='*60}")
    logging.info("Dual-blur generation complete")
    logging.info(f"{'='*60}")
    logging.info(f"succeeded: {success_count} images")
    if fail_count > 0:
        logging.info(f"failed: {fail_count} images")
    logging.info("\nOutput directories:")
    logging.info(f"   - Saliency-guided blur: {saliency_guided_dir}")
    logging.info(f"   - Center-guided blur: {center_guided_dir}")
    if save_saliency_maps:
        logging.info(f"   - Saliency map: {saliency_maps_dir}")
    logging.info("\n Configuration parameters:")
    logging.info(f"   - Saliency-guided blur: {saliency_config}")
    logging.info(f"   - Center-guided blur: {center_config}")
    logging.info(f"{'='*60}\n")
    if fail_count:
        raise RuntimeError(f'Dual-blur generation failed for {fail_count} images')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True, help='Dataset root containing Image_set_Resize')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--saliency-backend', choices=['auto', 'u2net', 'fine-grained', 'spectral-residual'], default='auto')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--save-saliency-maps', action='store_true')
    parser.add_argument('--limit', type=int, help='Process at most this many missing image pairs')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    generate_dual_blur_images(
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir) if args.output_dir else None,
        use_gpu=not args.cpu,
        force_regenerate=args.force,
        save_saliency_maps=args.save_saliency_maps,
        saliency_backend=args.saliency_backend,
        limit=args.limit
    )
