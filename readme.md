# CAIA

Code for **Neural-Visual Decoding via Cognitive-guided Adaptive Blurring and Information-Constrained Alignment**.

## Environment

Python 3.10 and an NVIDIA GPU are required.

```bash
conda create -n caia python=3.10
conda activate caia
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

## Prepare the data

Download and extract [THINGS-EEG](https://huggingface.co/datasets/Haitao999/things-eeg) and [THINGS-MEG](https://huggingface.co/datasets/Haitao999/things-meg):

```text
data/
  things-eeg/
    Preprocessed_data_250Hz_whiten/
      sub-01/train.pt
      sub-01/test.pt
      ...                              # sub-01 through sub-10
    Image_set_Resize/
      train_images/<category>/<image>.jpg
      test_images/<category>/<image>.jpg
    Dual_Blur_Images/
      saliency_guided/{train_images,test_images}/<category>/<image>.npy
      center_guided/{train_images,test_images}/<category>/<image>.npy
  things-meg/
    Preprocessed_data/
      sub-01/train.pt
      sub-01/test.pt
      ...                              # sub-01 through sub-04
    Image_set_Resize/
      train_images/<category>/<image>.jpg
      test_images/<category>/<image>.jpg
    Dual_Blur_Images/
      saliency_guided/{train_images,test_images}/<category>/<image>.npy
      center_guided/{train_images,test_images}/<category>/<image>.npy
```

## Generate blurred images

Skip this step if both sets of dual-blur arrays already exist. Otherwise, run:

```bash
python base/generate_dual_blur.py --data-dir data/things-eeg --saliency-backend fine-grained --cpu
python base/generate_dual_blur.py --data-dir data/things-meg --saliency-backend fine-grained --cpu
```

Existing image pairs are skipped. Add `--force` to regenerate them or `--limit 2` for a small preprocessing check.

## Run experiments

Run commands from the repository root:

```bash
# EEG, within subject
python main.py --dataset eeg --subjects sub-08 --epoch 50 --lr 1e-4

# MEG, within subject
python main.py --dataset meg --subjects sub-01 --epoch 50 --lr 5e-5

# EEG, leave one subject out
python main.py --dataset eeg --subjects sub-08 --exp_setting inter-subject --epoch 50 --lr 1e-5

# MEG, leave one subject out
python main.py --dataset meg --subjects sub-01 --exp_setting inter-subject --epoch 50 --lr 1e-5
```

The default training batch size is 128 and model is RN50.

On Windows, activate the environment and launch all subjects with:

```bat
scripts\train_eeg.bat
scripts\train_meg.bat
scripts\train_eeg.bat inter-subject
scripts\train_meg.bat inter-subject
```

## Evaluate pretrained weights

| Checkpoint | Download |
|---|---|
| EEG sub-08 | [Download](https://huggingface.co/yinfan233/CAIA/resolve/main/eeg-sub-08-intra.ckpt) |

```bash
python main.py --dataset eeg --subjects sub-08 --eval-only --checkpoint weights/eeg-sub-08-intra.ckpt
```

## Results

Training automatically saves and evaluates the best Top-1, best Top-5, and final checkpoints. Results are written to `all_test_results.json`, with TensorBoard logs and configurations in the same experiment directory.

Default output directories are `exp/caia` for EEG and `exp/MEG/caia` for MEG.

## License

[Apache License 2.0](LICENSE). Dataset and pretrained model licenses remain with their respective owners.
