import os
import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from pathlib import Path
from base.data_common import DualBlurCacheMixin, build_neural_loaders, build_test_loader


# Custom batch collation
def custom_collate_fn(batch):
    """Stack neural signals and image pairs into a batch."""
    # Collect sample fields
    eeg = torch.stack([item['eeg'] for item in batch])
    eeg_mean = torch.stack([item['eeg_mean'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])

    # Return the two blurred batches as a tuple
    blur_a_list = []
    blur_b_list = []
    for item in batch:
        blur_tuple = item['blur_paths']
        if isinstance(blur_tuple, (tuple, list)):
            blur_a_list.append(blur_tuple[0])
            blur_b_list.append(blur_tuple[1])
        else:
            raise TypeError(f"blur_paths Invalid format: {type(blur_tuple)}")

    blur_a = torch.stack(blur_a_list)
    blur_b = torch.stack(blur_b_list)
    blur_paths = (blur_a, blur_b)  # Tuple format

    # Additional sample fields
    texts = [item['text'] for item in batch]
    img_paths = [item['img_path'] for item in batch]
    sessions = torch.from_numpy(np.array([item['session'] for item in batch]))
    subjects = torch.from_numpy(np.array([item['subject'] for item in batch]))
    indices = torch.from_numpy(np.array([item['idx'] for item in batch]))

    return {
        'eeg': eeg,
        'eeg_mean': eeg_mean,
        'label': labels,
        'blur_paths': blur_paths,
        'text': texts,
        'img_path': img_paths,
        'session': sessions,
        'subject': subjects,
        'idx': indices,
    }


def load_eeg_data(config):
    """Build loaders with training-only held-out validation categories."""
    return build_neural_loaders(config, EEGDataset, custom_collate_fn, subject_count=10)


def load_eeg_test_data(config):
    """Build a test-only loader without accessing the training files."""
    return build_test_loader(config, EEGDataset, custom_collate_fn, subject_count=10)


class EEGDataset(DualBlurCacheMixin, Dataset):
    """Load preprocessed EEG trials and cache their dual-blur image pairs."""

    def __init__(self, config, mode):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                        'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                        'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                        'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                        'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                        'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                        'O1', 'Oz', 'O2']

        if self.selected_ch == "None":
            self.selected_ch = self.channels

        self.avg = config['data'][f"{mode}_avg"]
        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode == 'train' else 200
        self.per_trials = 4 if self.mode == 'train' else 80

        # ==================== Load EEG data ====================
        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        self.loaded_data = [self.load_data(data_path) for data_path in self.data_paths]

        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)

        # ==================== Dual-blur image paths ====================
        self.dual_blur_dir = Path(self.data_dir) / '../Dual_Blur_Images'
        self.saliency_blur_dir = self.dual_blur_dir / 'saliency_guided' / (f'{mode}_images')
        self.center_blur_dir = self.dual_blur_dir / 'center_guided' / (f'{mode}_images')

        if not self.saliency_blur_dir.exists() or not self.center_blur_dir.exists():
            raise FileNotFoundError(
                f"Dual-blur image directories do not exist!\n"
                f"Saliency-guided blur: {self.saliency_blur_dir}\n"
                f"Center-guided blur: {self.center_blur_dir}\n"
                f"Run python base/generate_dual_blur.py first"
            )

        # Initialize the image-pair cache
        self.saliency_images = {}
        self.center_images = {}

        logging.info(f"Initializing dual-blur image cache (mode={mode})")
        self._initialize_blur_cache()

        logging.info(f"EEGDataset initialized (mode={mode})")
        logging.info(f"  - Sample count: {len(self)}")
        logging.info(f"  - Preloaded blur pairs: {len(self.saliency_images)}")


    def load_data(self, data_path):
        """Load neural trials, select channels, and average or flatten repetitions."""
        logging.info(f"----load {data_path}----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])

        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]

        if self.avg:
            avg_data = {}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label'][:, 0]
            avg_data['img'] = loaded_data['img'][:, 0]
            avg_data['text'] = loaded_data['text'][:, 0]
            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = loaded_data['text'].reshape(-1)
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            loaded_data = _data

        return loaded_data

    # ==================== Dataset access ====================

    def __getitem__(self, index):
        """Return a trial, its temporal crop, metadata, and blurred image pair."""
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index // self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]
        text = self.loaded_data[subject]['text'][trial_index]
        session = self.loaded_data[subject]['session'][trial_index]

        blur_a, blur_b = self._load_blur_pair(img_path)

        return {
            'idx': index,
            'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
            'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
            'blur_paths': (blur_a, blur_b),
            'label': label,
            'text': f"This is a {text}.",
            'img_path': img_path,
            'session': session,
            'subject': subject,
        }

    def __len__(self):
        return self.trial_all_subjects
