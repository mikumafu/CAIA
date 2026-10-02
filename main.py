import argparse
import os
from pathlib import Path
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import shutil
import json
import pytorch_lightning as pl
from torch.optim import AdamW, Adam, SGD
import numpy as np
import logging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from base.data_eeg import load_eeg_data, load_eeg_test_data
from base.data_meg import load_meg_data, load_meg_test_data
from base.utils import update_config, ClipLoss, instantiate_from_config, load_blur_images
from base.cross_modal_attention import CrossModalAttention, BoundaryDistanceLoss

torch.set_float32_matmul_precision('medium')


def load_model(config):
    """Instantiate the configured models and training module."""
    model = {}
    for k, v in config['models'].items():
        print(f"init {k}")
        model[k] = instantiate_from_config(v)
    pl_model = PLModel(model, config)
    return pl_model


class PLModel(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.automatic_optimization = False
        self.config = config

        for key, value in model.items():
            setattr(self, f"{key}", value)

        # Attention module
        eeg_dim = config.get('z_dim', 1024)
        attention_config = dict(config.get('attention', {}))
        attention_config.setdefault('eeg_dim', eeg_dim)
        attention_config.setdefault('img_channels', 3)
        self.attention_module = CrossModalAttention(**attention_config)
        logging.info(f"Initialize CrossModalAttention (eeg_dim={eeg_dim})")


        import open_clip
        vision_backbone = config['data']['model_type']
        pretrained = config['data'].get('pretrained', 'openai')
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            vision_backbone, pretrained=pretrained
        )
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True
        self.clip_model.train()
        logging.info(f"Loaded CLIP visual encoder: {vision_backbone} ({pretrained})")

        # Loss functions
        self.criterion = ClipLoss()
        self.boundary_loss_fn = BoundaryDistanceLoss(alpha=config.get('loss', {}).get('boundary_alpha', 0.05))
        self.boundary_weight = config['train'].get('boundary_weight', 0.01)
        print(f"Boundary loss weight: {self.boundary_weight}")

        self.running_mean = 0.0
        self.running_std = 1.0
        self.warmup_epochs = config['train'].get('warmup_epochs', 0)

        self.all_predicted_classes = []
        self.all_true_labels = []
        self.mAP_total = 0
        self.match_similarities = []
        self.denoise_info = {}

    def forward(self, batch):
        eeg = batch['eeg']
        blur_paths = batch['blur_paths']

        # 1. Encode EEG
        if self.training and self.config.get('use_denoising', True):
            eeg_z, denoise_info = self.brain(eeg, return_denoise_info=True)
        else:
            eeg_z = self.brain(eeg, return_denoise_info=False)
            denoise_info = {}

        # 2. Encode images
        blur_a, blur_b = load_blur_images(blur_paths, device=self.device)
        fused_img, attn_map = self.attention_module(eeg_z, blur_a, blur_b)
        img_z = self.clip_model.encode_image(fused_img)

        # 3. L2 Normalization
        img_z = img_z / img_z.norm(dim=-1, keepdim=True)
        eeg_z = eeg_z / eeg_z.norm(dim=-1, keepdim=True)

        # 4. Contrastive loss
        logit_scale = self.brain.logit_scale
        logit_scale = self.brain.softplus(logit_scale)
        eeg_loss, img_loss, logits_per_image = self.criterion(eeg_z, img_z, logit_scale)
        clip_loss = (eeg_loss.mean() + img_loss.mean()) / 2

        # 5. Boundary loss during training
        boundary_loss = None
        if self.training:
            diagonal_sim = torch.diagonal(logits_per_image).detach().cpu().numpy()
            batch_mean = diagonal_sim.mean()
            batch_std = diagonal_sim.std(ddof=1) if len(diagonal_sim) > 1 else 1.0
            alpha = 0.1
            self.running_mean = alpha * batch_mean + (1 - alpha) * self.running_mean
            self.running_std = alpha * batch_std + (1 - alpha) * self.running_std
            running_stats = {'mean': self.running_mean, 'std': self.running_std}
            boundary_loss, abnormal_mask = self.boundary_loss_fn(
                logits_per_image, batch['idx'], running_stats
            )
            if len(abnormal_mask) > 0:
                abnormal_ratio = abnormal_mask.float().mean()
            else:
                abnormal_ratio = torch.tensor(0.0, device=self.device)
            self.log('abnormal_ratio', abnormal_ratio, on_step=False, on_epoch=True)

        self.denoise_info = denoise_info
        return eeg_z, img_z, clip_loss, boundary_loss, attn_map

    def training_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        opt = self.optimizers()
        eeg_z, img_z, clip_loss, boundary_loss, attn_map = self(batch)

        if self.current_epoch < self.warmup_epochs:
            boundary_loss = None

        ib_loss = torch.tensor(0.0, device=self.device)
        if self.training and hasattr(self, 'denoise_info') and 'ib_loss' in self.denoise_info:
            ib_loss_value = self.denoise_info['ib_loss']
            if isinstance(ib_loss_value, float):
                ib_loss = torch.tensor(ib_loss_value, device=self.device)
            else:
                ib_loss = ib_loss_value

        if boundary_loss is not None and self.boundary_weight != 0.0:
            total_loss = clip_loss + self.boundary_weight * boundary_loss + ib_loss
            self.log('boundary_loss', boundary_loss, on_step=True, on_epoch=True,
                     prog_bar=False, batch_size=batch_size)
        else:
            total_loss = clip_loss + ib_loss

        self.log('ib_loss', ib_loss, on_step=True, on_epoch=True,
                 prog_bar=False, batch_size=batch_size)

        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()

        self.log('train_loss', total_loss, on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('clip_loss', clip_loss, on_step=True, on_epoch=True,
                 prog_bar=False, batch_size=batch_size)

        similarity = (eeg_z @ img_z.T)
        top_k_indices = similarity.topk(min(5, batch_size), dim=-1).indices
        # Pad small final batches with an invalid candidate index.
        predictions = top_k_indices.cpu().numpy()
        predictions = np.pad(predictions, ((0, 0), (0, 5 - predictions.shape[1])), constant_values=-1)
        self.all_predicted_classes.append(predictions)
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())

        if batch_idx == self.trainer.num_training_batches - 1:
            all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
            all_true_labels = np.array(self.all_true_labels)
            top_1_correct = (all_predicted_classes[:, 0] == all_true_labels)
            top_1_accuracy = top_1_correct.mean()
            top_k_correct = (all_predicted_classes == all_true_labels[:, None]).any(axis=1)
            top_k_accuracy = top_k_correct.mean()
            self.log('train_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            self.log('train_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,
                     prog_bar=True, logger=True, sync_dist=True)
            self.all_predicted_classes = []
            self.all_true_labels = []

        return total_loss

    def validation_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, clip_loss, boundary_loss, _ = self(batch)
        if boundary_loss is not None:
            loss = clip_loss + self.boundary_weight * boundary_loss
        else:
            loss = clip_loss
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=batch_size)
        similarity = (eeg_z @ img_z.T)
        top_k_indices = similarity.topk(min(5, batch_size), dim=-1).indices
        # Pad small final batches with an invalid candidate index.
        predictions = top_k_indices.cpu().numpy()
        predictions = np.pad(predictions, ((0, 0), (0, 5 - predictions.shape[1])), constant_values=-1)
        self.all_predicted_classes.append(predictions)
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())
        return loss

    def on_validation_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_accuracy = (all_predicted_classes[:, 0] == all_true_labels).mean()
        top_k_accuracy = (all_predicted_classes == all_true_labels[:, None]).any(axis=1).mean()
        self.log('val_top1_acc', top_1_accuracy, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log('val_top5_acc', top_k_accuracy, on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []

    def test_step(self, batch, batch_idx):
        batch_size = batch['idx'].shape[0]
        eeg_z, img_z, clip_loss, boundary_loss, _ = self(batch)
        if boundary_loss is not None:
            loss = clip_loss + self.boundary_weight * boundary_loss
        else:
            loss = clip_loss
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True, batch_size=batch_size)
        similarity = (eeg_z @ img_z.T)
        top_k_indices = similarity.topk(min(5, batch_size), dim=-1).indices
        # Pad small final batches with an invalid candidate index.
        predictions = top_k_indices.cpu().numpy()
        predictions = np.pad(predictions, ((0, 0), (0, 5 - predictions.shape[1])), constant_values=-1)
        self.all_predicted_classes.append(predictions)
        label = torch.arange(0, batch_size).to(self.device)
        self.all_true_labels.extend(label.cpu().numpy())
        self.match_similarities.extend(similarity.diag().detach().cpu().tolist())
        for i in range(similarity.shape[0]):
            true_index = i
            sims = similarity[i, :]
            sorted_indices = torch.argsort(-sims)
            rank = (sorted_indices == true_index).nonzero()[0][0] + 1
            ap = 1 / rank
            self.mAP_total += ap
        return loss

    def on_test_epoch_end(self):
        all_predicted_classes = np.concatenate(self.all_predicted_classes, axis=0)
        all_true_labels = np.array(self.all_true_labels)
        top_1_accuracy = (all_predicted_classes[:, 0] == all_true_labels).mean()
        top_k_accuracy = (all_predicted_classes == all_true_labels[:, None]).any(axis=1).mean()
        self.mAP = (self.mAP_total / len(all_true_labels)).item()
        self.match_similarities = np.mean(self.match_similarities) if self.match_similarities else 0
        self.log('test_top1_acc', top_1_accuracy, sync_dist=True)
        self.log('test_top5_acc', top_k_accuracy, sync_dist=True)
        self.log('mAP', self.mAP, sync_dist=True)
        self.log('similarity', self.match_similarities, sync_dist=True)
        self.all_predicted_classes = []
        self.all_true_labels = []
        self.mAP_total = 0
        self.match_similarities = []

    def configure_optimizers(self):
        params_to_optimize = [
            {'params': self.brain.parameters(), 'lr': self.config['train']['lr']},
            {'params': self.attention_module.parameters(), 'lr': self.config['train']['lr']},
            {'params': self.clip_model.visual.parameters(), 'lr': self.config['train']['lr']},
        ]
        optimizer_class = {'AdamW': AdamW, 'Adam': Adam, 'SGD': SGD}[self.config['train']['optimizer']]
        optimizer = optimizer_class(
            params_to_optimize, weight_decay=self.config['train'].get('weight_decay', 1e-4)
        )
        # Manual optimization uses a constant learning rate.
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        checkpoint['attention_state_dict'] = self.attention_module.state_dict()
        checkpoint['mode'] = 'finetune_clip'
        checkpoint['running_mean'] = self.running_mean
        checkpoint['running_std'] = self.running_std
        logging.info("Save checkpoint (mode=finetune_clip)")

    def on_load_checkpoint(self, checkpoint):
        self.running_mean = checkpoint.get('running_mean', 0.0)
        self.running_std = checkpoint.get('running_std', 1.0)


def evaluate_checkpoint(config, checkpoint_path):
    """Evaluate saved weights without training or downloading pretrained weights."""
    evaluation_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    evaluation_config['data']['pretrained'] = None
    test_loader = (load_eeg_test_data(evaluation_config) if config['dataset'] == 'eeg'
                   else load_meg_test_data(evaluation_config))
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = load_model(evaluation_config)
    model.load_state_dict(checkpoint.get('state_dict', checkpoint), strict=True)
    model.on_load_checkpoint(checkpoint)
    del checkpoint
    logger = TensorBoardLogger(
        config['save_dir'], name=config['name'],
        version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}_evaluation",
    )
    trainer = Trainer(devices=1, accelerator='cuda', logger=logger,
                      enable_checkpointing=False, enable_progress_bar=True)
    results = trainer.test(model, dataloaders=test_loader)[0]
    results_file = Path(logger.log_dir) / 'checkpoint_test_results.json'
    with results_file.open('w', encoding='utf-8') as stream:
        json.dump({'model_path': str(checkpoint_path), 'test_results': results}, stream, indent=4)
    print(f'Checkpoint results saved to: {results_file}')
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="eeg", choices=["eeg", "meg"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subjects", type=str, default='sub-01')
    parser.add_argument("--exp_setting", choices=["intra-subject", "inter-subject"], default="intra-subject")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--brain_backbone", type=str, default="EEGProjectLayer")
    parser.add_argument("--vision_backbone", type=str, default="RN50")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--data-root", type=Path, help="Directory containing things-eeg and things-meg")
    parser.add_argument("--save-dir", type=Path, help="Override the experiment output directory")
    parser.add_argument("--clip-pretrained", help="OpenCLIP pretrained tag or local checkpoint path")
    parser.add_argument("--batch-size", type=int, help="Override training batch size")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run one training, validation, and test batch")
    parser.add_argument("--preload-blur-images", action="store_true", help="Preload all image pairs instead of using a bounded cache")
    parser.add_argument("--resume", type=Path, help="Resume training from a saved Lightning checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a checkpoint without training")
    parser.add_argument("--checkpoint", type=Path, help="Model checkpoint for --eval-only")

    opt = parser.parse_args()
    if opt.eval_only != (opt.checkpoint is not None):
        parser.error('--eval-only and --checkpoint must be supplied together')
    if opt.eval_only and (opt.resume or opt.fast_dev_run):
        parser.error('--eval-only cannot be combined with --resume or --fast-dev-run')
    if opt.checkpoint is not None and not opt.checkpoint.is_file():
        parser.error(f'Checkpoint does not exist: {opt.checkpoint}')
    seed_everything(opt.seed)
    project_dir = Path(__file__).resolve().parent
    config_path = Path(opt.config) if opt.config else project_dir / "configs" / opt.dataset / "caia.yaml"
    if not config_path.is_absolute():
        config_path = project_dir / config_path
    config = OmegaConf.load(config_path)
    config = update_config(opt, config)
    config['data']['subjects'] = [opt.subjects]
    config['preload_blur_images'] = opt.preload_blur_images
    config['data']['num_workers'] = opt.num_workers
    config['data']['pin_memory'] = opt.pin_memory
    config['data']['persistent_workers'] = False

    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }
    if opt.vision_backbone not in pretrain_map:
        parser.error(f'Unsupported vision backbone: {opt.vision_backbone}')
    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']
    if opt.clip_pretrained:
        config['data']['pretrained'] = opt.clip_pretrained
    elif opt.vision_backbone != 'RN50':
        config['data']['pretrained'] = pretrain_map[opt.vision_backbone]['pretrained']
    if opt.data_root:
        preprocessing = 'Preprocessed_data_250Hz_whiten' if opt.dataset == 'eeg' else 'Preprocessed_data'
        config['data']['data_dir'] = str(opt.data_root.resolve() / f'things-{opt.dataset}' / preprocessing)
    else:
        data_path = Path(config['data']['data_dir'])
        if not data_path.is_absolute():
            config['data']['data_dir'] = str(project_dir / data_path)
    if opt.save_dir:
        config['save_dir'] = str(opt.save_dir.resolve())
    elif not Path(config['save_dir']).is_absolute():
        config['save_dir'] = str(project_dir / config['save_dir'])
    if opt.batch_size is not None:
        if opt.batch_size < 1:
            parser.error('--batch-size must be positive')
        config['data']['train_batch_size'] = opt.batch_size

    print("=" * 60)
    print("Configuration:")
    print(f"- CLIP visual encoder: {opt.vision_backbone}")
    print(f"- Subject: {config['data']['subjects']}")
    print("=" * 60)

    if opt.eval_only:
        evaluate_checkpoint(config, opt.checkpoint.resolve())
        return

    os.makedirs(config['save_dir'], exist_ok=True)
    logger = TensorBoardLogger(
        config['save_dir'], name=config['name'],
        version=f"{'_'.join(config['data']['subjects'])}_seed{config['seed']}"
    )
    os.makedirs(logger.log_dir, exist_ok=True)
    shutil.copy(config_path, Path(logger.log_dir) / config_path.name)
    OmegaConf.save(config, Path(logger.log_dir) / 'resolved_config.yaml', resolve=True)

    torch.cuda.empty_cache()

    train_loader, val_loader, test_loader = load_eeg_data(config) if config['dataset'] == 'eeg' else load_meg_data(config)
    with (Path(logger.log_dir) / 'data_split.json').open('w', encoding='utf-8') as stream:
        json.dump(train_loader.dataset.split_metadata, stream, indent=2)

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    pl_model = load_model(config)

    checkpoint_callback_top1 = ModelCheckpoint(
        monitor='val_top1_acc', mode='max', save_top_k=1,
        dirpath=logger.log_dir, filename='best_top1-{epoch:02d}-{val_top1_acc:.4f}',
        verbose=True, save_last=False
    )
    checkpoint_callback_top5 = ModelCheckpoint(
        monitor='val_top5_acc', mode='max', save_top_k=1,
        dirpath=logger.log_dir, filename='best_top5-{epoch:02d}-{val_top5_acc:.4f}',
        verbose=True, save_last=False
    )
    checkpoint_callback_last = ModelCheckpoint(
        save_last=True, dirpath=logger.log_dir, filename='last-{epoch:02d}', verbose=True
    )

    if config['exp_setting'] == 'inter-subject':
        early_stop_callback = EarlyStopping(monitor='val_top1_acc', min_delta=0.001,
                                            patience=45, verbose=False, mode='max')
    else:
        early_stop_callback = EarlyStopping(monitor='train_loss', min_delta=0.001,
                                            patience=45, verbose=False, mode='min')

    trainer = Trainer(
        log_every_n_steps=10, strategy='auto',
        callbacks=[checkpoint_callback_top1, checkpoint_callback_top5, checkpoint_callback_last, early_stop_callback],
        max_epochs=config['train']['epoch'], devices=1, accelerator='cuda',
        fast_dev_run=opt.fast_dev_run,
        logger=logger, enable_progress_bar=True
    )
    print(f"Log directory: {trainer.logger.log_dir}")

    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader,
                ckpt_path=str(opt.resume.resolve()) if opt.resume else None)

    # ========== Evaluate the three checkpoints ==========
    print("\n" + "="*80)
    print("Testing Phase - Evaluating 3 different models")
    print("="*80)

    if opt.fast_dev_run:
        trainer.test(pl_model, dataloaders=test_loader)
        return

    all_test_results = {}

    checkpoints = [
        ('best_top1_model', checkpoint_callback_top1.best_model_path,
         'best_val_top1_acc', checkpoint_callback_top1.best_model_score),
        ('best_top5_model', checkpoint_callback_top5.best_model_path,
         'best_val_top5_acc', checkpoint_callback_top5.best_model_score),
        ('last_model', checkpoint_callback_last.last_model_path, None, None),
    ]
    for name, path, score_name, score in checkpoints:
        print(f"\nEvaluating {name}: {path}")
        results = trainer.test(ckpt_path=path, dataloaders=test_loader)[0]
        entry = {'model_path': path, 'test_results': results}
        if score_name is not None:
            entry[score_name] = float(score)
        all_test_results[name] = entry

    results_file = Path(logger.log_dir) / 'all_test_results.json'
    with results_file.open('w', encoding='utf-8') as stream:
        json.dump(all_test_results, stream, indent=4)
    for name, entry in all_test_results.items():
        print(f"\n{name}:")
        for key, value in entry['test_results'].items():
            print(f"   {key}: {value:.4f}" if isinstance(value, float) else f"   {key}: {value}")
    print(f"\nAll results saved to: {results_file}")


if __name__ == "__main__":
    main()
