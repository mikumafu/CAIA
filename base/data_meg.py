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

    if 'label' in batch[0]:
        labels = torch.tensor([item['label'] for item in batch])
    else:
        labels = torch.arange(len(batch))  # Use batch indices when labels are absent

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
    blur_paths = (blur_a, blur_b)

    # Additional sample fields
    texts = [item.get('text', '') for item in batch]
    img_paths = [item['img_path'] for item in batch]

    if 'session' in batch[0]:
        sessions = torch.from_numpy(np.array([item['session'] for item in batch]))
    else:
        sessions = torch.zeros(len(batch), dtype=torch.long)

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


def load_meg_data(config):
    """Build loaders with training-only held-out validation categories."""
    return build_neural_loaders(config, MEGDataset, custom_collate_fn, subject_count=4)


def load_meg_test_data(config):
    """Build a test-only loader without accessing the training files."""
    return build_test_loader(config, MEGDataset, custom_collate_fn, subject_count=4)


class MEGDataset(DualBlurCacheMixin, Dataset):
    """Load preprocessed MEG trials and cache their dual-blur image pairs."""

    def __init__(self, config, mode):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = None

        if self.selected_ch == "None" or self.selected_ch == False:
            self.selected_ch = None

        self.avg = config['data'][f"{mode}_avg"]
        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode == 'train' else 200
        self.per_trials = 1 if self.mode == 'train' else 12

        # ==================== Load MEG data ====================
        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        self.loaded_data = [self.load_data(data_path) for data_path in self.data_paths]

        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)

        # ==================== Dual-blur image paths ====================
        self.dual_blur_dir = Path(self.data_dir).parent / 'Dual_Blur_Images'
        self.saliency_blur_dir = self.dual_blur_dir / 'saliency_guided' / f'{mode}_images'
        self.center_blur_dir = self.dual_blur_dir / 'center_guided' / f'{mode}_images'

        # Check required directories
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

        logging.info(f"MEGDataset initialized (mode={mode})")
        logging.info(f"  - Sample count: {len(self)}")
        logging.info(f"  - Preloaded blur pairs: {len(self.saliency_images)}")


    def load_data(self, data_path):
        """Load neural trials, select channels, and average or flatten repetitions."""
        logging.info(f"----load {data_path}----")
        loaded_data = torch.load(data_path, weights_only=False)
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])


        if self.selected_ch and self.channels is not None:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]


        if self.avg:
            avg_data = {}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['img'] = np.array(loaded_data['img'])

            if 'label' in loaded_data:
                avg_data['label'] = loaded_data['label'][:, 0] if loaded_data['label'].ndim > 1 else loaded_data['label']
            if 'text' in loaded_data:
                avg_data['text'] = loaded_data['text'][:, 0] if loaded_data['text'].ndim > 1 else loaded_data['text']
            if 'session' in loaded_data:
                avg_data['session'] = loaded_data['session']
            if 'times' in loaded_data:
                avg_data['times'] = loaded_data['times']

            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['img'] = loaded_data['img'].reshape(-1)

            if 'label' in loaded_data:
                _data['label'] = loaded_data['label'].reshape(-1)
            if 'text' in loaded_data:
                _data['text'] = loaded_data['text'].reshape(-1)
            if 'session' in loaded_data:
                _data['session'] = loaded_data['session'].reshape(-1)
            if 'times' in loaded_data:
                _data['times'] = loaded_data['times']

            loaded_data = _data

        print("Data shape", loaded_data['eeg'].shape)

        for k, v in loaded_data.items():
            if k in ['eeg', 'label', 'img', 'text', 'session']:
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    logging.info(f"{k}: {v.shape}")
                else:
                    logging.info(f"{k}: {len(v)} items")

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

        img_path = self.loaded_data[subject]['img'][trial_index]

        data = self.loaded_data[subject]
        label = data['label'][trial_index] if 'label' in data else trial_index
        text = data['text'][trial_index] if 'text' in data else 'unknown'
        session = data['session'][trial_index] if 'session' in data else 0

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
