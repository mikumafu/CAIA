"""Shared dataset splitting, loaders, and dual-blur caching for EEG and MEG."""

import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict
import copy

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset


def image_category(img_path):
    """Return a category identifier independent of path separator or split prefix."""
    parts = Path(str(img_path).replace('\\', '/')).parts
    if len(parts) != 3:
        raise ValueError(f'Expected split/category/image path, got: {img_path}')
    split, category, _ = parts
    return category[len(split) + 1:] if category.startswith(split + '_') else category


class ValidationSubset(Subset):
    """Evaluate one averaged stimulus per held-out category and subject."""
    def __getitem__(self, index):
        sample = self.dataset[self.indices[index]]
        sample['eeg'] = sample['eeg_mean']
        return sample

    def __getitems__(self, indices):
        return [self[index] for index in indices]


def split_training_categories(dataset, validation_classes=200, seed=0):
    """Reserve whole categories for zero-shot validation using only training data."""
    subject_paths = [list(data['img']) for data in dataset.loaded_data]
    categories = sorted({image_category(path) for path in subject_paths[0]})
    if not 1 < validation_classes < len(categories):
        raise ValueError('validation_classes must be at least 2 and leave training categories available')
    validation_categories = set(np.random.default_rng(seed).choice(categories, validation_classes, replace=False).tolist())
    train_indices, validation_indices = [], []
    offset = 0
    for paths in subject_paths:
        if {image_category(path) for path in paths} != set(categories):
            raise ValueError('Training subjects must have the same stimulus categories')
        representatives = {}
        for index, path in enumerate(paths):
            category = image_category(path)
            if category in validation_categories:
                candidate = (str(path), offset + index)
                if category not in representatives or candidate < representatives[category]:
                    representatives[category] = candidate
            else:
                train_indices.append(offset + index)
        validation_indices.extend(representatives[category][1] for category in sorted(validation_categories))
        offset += len(paths)
    metadata = {
        'validation_seed': seed,
        'training_categories': sorted(set(categories) - validation_categories),
        'validation_categories': sorted(validation_categories),
        'training_subjects': list(dataset.subjects),
        'validation_subjects': list(dataset.subjects),
        'training_samples': len(train_indices),
        'validation_samples': len(validation_indices),
    }
    train, val = Subset(dataset, train_indices), ValidationSubset(dataset, validation_indices)
    train.split_metadata = val.split_metadata = metadata
    return train, val


def build_neural_loaders(config, dataset_class, collate_fn, subject_count):
    """Construct independent training, validation, and final-test loaders."""
    setting = config.get('exp_setting', 'intra-subject')
    data_config = config['data']
    test_subjects = list(data_config['subjects'])
    all_subjects = [f'sub-{index:02}' for index in range(1, subject_count + 1)]
    if not test_subjects or not set(test_subjects).issubset(all_subjects):
        raise ValueError(f'Invalid subjects: {test_subjects}; expected a subset of {all_subjects}')
    training_config = copy.deepcopy(config)
    if setting == 'inter-subject':
        training_config['data']['subjects'] = [subject for subject in all_subjects if subject not in test_subjects]
    elif setting != 'intra-subject':
        raise ValueError(f'Unsupported experiment setting: {setting}')
    if not training_config['data']['subjects']:
        raise ValueError('At least one training subject must remain')

    test_dataset = dataset_class(config, mode='test')
    training_pool = dataset_class(training_config, mode='train')
    validation_classes = data_config.get('validation_classes', 200)
    train_dataset, val_dataset = split_training_categories(
        training_pool, validation_classes=validation_classes,
        seed=data_config.get('validation_seed', 0),
    )
    test_categories = {image_category(path) for data in test_dataset.loaded_data for path in data['img']}
    train_categories = set(train_dataset.split_metadata['training_categories'])
    val_categories = set(train_dataset.split_metadata['validation_categories'])
    if test_categories & (train_categories | val_categories):
        raise ValueError('Test categories overlap the original training pool')
    train_dataset.split_metadata.update(test_subjects=test_subjects,
                                        test_categories=sorted(test_categories), test_samples=len(test_dataset))
    workers = data_config.get('num_workers', 0)
    common = dict(num_workers=workers, pin_memory=data_config.get('pin_memory', True),
                  persistent_workers=workers > 0 and data_config.get('persistent_workers', True),
                  collate_fn=collate_fn, drop_last=False)
    train_loader = DataLoader(train_dataset, batch_size=data_config['train_batch_size'], shuffle=True, **common)
    val_loader = DataLoader(val_dataset, batch_size=min(data_config['val_batch_size'], validation_classes), shuffle=False, **common)
    test_loader = DataLoader(test_dataset, batch_size=data_config['test_batch_size'], shuffle=False, **common)
    return train_loader, val_loader, test_loader


def build_test_loader(config, dataset_class, collate_fn, subject_count):
    """Load only the requested subjects' test data for checkpoint evaluation."""
    data_config = config['data']
    subjects = list(data_config['subjects'])
    valid_subjects = {f'sub-{index:02}' for index in range(1, subject_count + 1)}
    if not subjects or not set(subjects).issubset(valid_subjects):
        raise ValueError(f'Invalid test subjects: {subjects}')
    workers = data_config.get('num_workers', 0)
    return DataLoader(
        dataset_class(config, mode='test'), batch_size=data_config['test_batch_size'],
        shuffle=False, num_workers=workers, pin_memory=data_config.get('pin_memory', True),
        persistent_workers=workers > 0 and data_config.get('persistent_workers', True),
        collate_fn=collate_fn, drop_last=False,
    )


class DualBlurCacheMixin:
    def _initialize_blur_cache(self):
        """Use a bounded cache by default; optionally preload every pair."""
        self.preload_blur_images = self.config.get('preload_blur_images', False)
        self.image_cache_size = self.config.get('image_cache_size', 128)
        self._image_cache = OrderedDict()
        if self.preload_blur_images:
            self._preload_all_blur_images()

    def _load_blur_pair(self, img_path):
        if self.preload_blur_images:
            return self.saliency_images[img_path], self.center_images[img_path]
        if img_path in self._image_cache:
            self._image_cache.move_to_end(img_path)
            return self._image_cache[img_path]
        paths = self._get_dual_blur_paths_from_img_path(img_path)
        pair = tuple(torch.from_numpy(np.load(path).astype('float32')) for path in paths)
        if self.image_cache_size > 0:
            self._image_cache[img_path] = pair
            if len(self._image_cache) > self.image_cache_size:
                self._image_cache.popitem(last=False)
        return pair

    def _preload_all_blur_images(self):
        """Load each unique blurred image pair once, using worker threads."""
        all_img_paths_set = set()
        for subject_idx in range(len(self.subjects)):
            for trial_idx in range(self.trial_subject):
                img_path = self.loaded_data[subject_idx]['img'][trial_idx]
                all_img_paths_set.add(img_path)

        all_img_paths = list(all_img_paths_set)
        total_images = len(all_img_paths)
        total_samples = len(self.subjects) * self.trial_subject
        logging.info(f"  - Unique images: {total_images}, Total samples: {total_samples} (across {len(self.subjects)} subjects)")

        def load_image_pair(img_path):
            saliency_path, center_path = self._get_dual_blur_paths_from_img_path(img_path)
            if not Path(saliency_path).exists():
                raise FileNotFoundError(f"Saliency-blurred image does not exist: {saliency_path}")
            if not Path(center_path).exists():
                raise FileNotFoundError(f"Center-blurred image does not exist: {center_path}")
            saliency_img = np.load(saliency_path).astype('float32')
            center_img = np.load(center_path).astype('float32')
            saliency_tensor = torch.from_numpy(saliency_img)
            center_tensor = torch.from_numpy(center_img)
            return img_path, saliency_tensor, center_tensor

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(load_image_pair, img_path) for img_path in all_img_paths]
            for future in tqdm(as_completed(futures), total=total_images, desc="Preloading blurred images"):
                img_path, saliency_tensor, center_tensor = future.result()
                self.saliency_images[img_path] = saliency_tensor
                self.center_images[img_path] = center_tensor

        loaded_count = len(self.saliency_images)
        if loaded_count != total_images:
            raise RuntimeError(f"Preloaded image count mismatch: expected {total_images}, actual {loaded_count}")

        logging.info(f"Successfully preloaded {loaded_count} unique image pairs into memory")
        if loaded_count > 0:
            sample_tensor = next(iter(self.saliency_images.values()))
            tensor_size_bytes = sample_tensor.element_size() * sample_tensor.numel()
            total_memory_mb = (2 * loaded_count * tensor_size_bytes) / (1024**2)
            logging.info(f"  - Estimated memory usage: {total_memory_mb:.2f} MB")

    def _get_dual_blur_paths_from_img_path(self, img_path):
        """Resolve split/category/image paths to the two cached NumPy arrays."""
        parts = Path(str(img_path).replace('\\', '/')).parts
        if len(parts) != 3:
            raise ValueError(f"Invalid image path format: {img_path}")

        mode, category, filename = parts

        if category.startswith(f'{mode}_'):
            category_clean = category[len(mode) + 1:]
        else:
            category_clean = category

        npy_filename = Path(filename).stem + '.npy'

        saliency_path = str(self.saliency_blur_dir / category_clean / npy_filename)
        center_path = str(self.center_blur_dir / category_clean / npy_filename)

        return (saliency_path, center_path)
