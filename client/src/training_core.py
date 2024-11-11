import asyncio
import os
import traceback

import torch
import torch.nn as nn
from typing import Dict, Optional, Any
import hashlib
import json
import requests
from dataclasses import dataclass
import logging
import importlib.util
import sys

logger = logging.getLogger("AIFarmTraining")


@dataclass
class TrainingConfig:
    model_name: str
    batch_size: int
    learning_rate: float
    epochs: int
    device: str
    distributed: bool = False
    num_workers: int = 4
    checkpoint_freq: int = 1


class CheckpointManager:
    def __init__(self, save_dir: str = "checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, metrics: Dict, task_id: str):
        path = os.path.join(self.save_dir, f"task_{task_id}_epoch_{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, path)
        return path

    def load_checkpoint(self, path: str) -> Dict:
        return torch.load(path)


class TrainingManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.checkpoint_manager = CheckpointManager()
        self.model = None
        self.optimizer = None
        self.current_epoch = 0
        self.progress_callback = None

        # Training metrics'i doğru initialize et
        self.training_metrics = {
            "loss": [],  # Array olarak başlat
            "accuracy": [],  # Array olarak başlat
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": config.epochs,
            "status": "initializing"
        }

    async def set_progress_callback(self, callback):
        self.progress_callback = callback

    async def update_progress(self, metrics: Dict):
        """Progress ve metrics'i güncelle ve callback'i çağır"""
        try:
            if isinstance(metrics, dict):
                # Özel metrik güncellemeleri
                if 'progress' in metrics:
                    self.training_metrics['progress'] = metrics['progress']
                if 'epoch' in metrics:
                    self.training_metrics['current_epoch'] = metrics['epoch']
                if 'loss' in metrics and isinstance(self.training_metrics['loss'], list):
                    if isinstance(metrics['loss'], (int, float)):
                        self.training_metrics['loss'].append(metrics['loss'])
                if 'accuracy' in metrics and isinstance(self.training_metrics['accuracy'], list):
                    if isinstance(metrics['accuracy'], (int, float)):
                        self.training_metrics['accuracy'].append(metrics['accuracy'])

                # Diğer metrikleri güncelle
                for key, value in metrics.items():
                    if key not in ['loss', 'accuracy']:  # Bu listeleri korumak için
                        self.training_metrics[key] = value

            if self.progress_callback:
                await self.progress_callback(self.training_metrics)

        except Exception as e:
            logger.error(f"Error updating progress: {str(e)}\n{traceback.format_exc()}")
    def train_step(self, batch):
        """Local training step that handles device placement"""
        data, target = batch
        data = data.to(self.device)
        target = target.to(self.device)

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = nn.functional.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
        loss.backward()
        self.optimizer.step()

        return loss.item()

    # training_core.py içindeki train metodunu güncelleyin

    async def train(self, task_id: str, model_code: str, train_data: Any, resume_from: Optional[str] = None) -> Dict:
        try:
            logger.info(f"\nStarting training for task {task_id}")
            logger.info(f"Configuration: {self.config.__dict__}")

            # Training metrics'i sıfırla
            self.training_metrics = {
                "loss": [],
                "accuracy": [],
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": self.config.epochs,
                "status": "initializing"
            }

            # Load model code and setup
            spec = importlib.util.spec_from_loader("dynamic_model", loader=None)
            module = importlib.util.module_from_spec(spec)
            sys.modules["dynamic_model"] = module
            exec(model_code, module.__dict__)
            logger.info("Model code loaded successfully")

            # Initialize model
            self.model, self.optimizer = module.create_model()
            self.model = self.model.to(self.device)
            logger.info(f"Model created and moved to device: {self.device}")

            # Shard information
            shard_info = train_data.get('shard_info', {})
            shard_id = shard_info.get('id', 0)
            start_point = shard_info.get('start_point', 0)
            end_point = shard_info.get('end_point', 100)
            logger.info(f"Training shard {shard_id}: {start_point}% - {end_point}%")



            # Create smaller dataset for testing
            total_data = torch.randint(0, 10, (10, 5))  # Küçük test dataset'i
            shard_size = len(total_data)
            start_idx = int(start_point * shard_size / 100)
            end_idx = int(end_point * shard_size / 100)
            shard_data = total_data[start_idx:end_idx]
            logger.info(f"Training on {end_idx - start_idx} samples")

            # Training setup
            best_loss = float('inf')
            self.training_metrics["status"] = "running"
            progress = 0

            # Training loop
            for epoch in range(self.current_epoch, self.config.epochs):
                epoch_loss = 0.0
                batch_count = 0
                self.training_metrics["current_epoch"] = epoch + 1

                # Batch training
                batch_size = self.config.batch_size
                n_batches = max(1, len(shard_data) // batch_size)

                for batch_idx in range(n_batches):
                    try:
                        # Get batch data
                        start = batch_idx * batch_size
                        end = min(start + batch_size, len(shard_data))
                        batch_data = shard_data[start:end]

                        # Create matching target data with same shape as input
                        vocab_size = self.model.config.vocab_size if hasattr(self.model, 'config') else 1000

                        # Create target tensor with matching dimensions
                        input_size = batch_data.size(1)  # Get the sequence length from input
                        batch_target = torch.randint(0, vocab_size, (len(batch_data), input_size))

                        # Log dimensions for debugging
                        logger.info(f"Input shape: {batch_data.shape}, Target shape: {batch_target.shape}")

                        # Ensure both tensors match in dimension
                        assert batch_data.size(0) == batch_target.size(
                            0), f"Batch size mismatch: {batch_data.size(0)} vs {batch_target.size(0)}"
                        assert batch_data.size(1) == batch_target.size(
                            1), f"Sequence length mismatch: {batch_data.size(1)} vs {batch_target.size(1)}"

                        # Train step
                        loss = self.train_step((batch_data, batch_target))
                        if loss is not None:  # Only count if training was successful
                            epoch_loss += loss
                            batch_count += 1

                        # Calculate progress
                        current_step = (epoch * n_batches) + batch_idx + 1
                        total_steps = self.config.epochs * n_batches
                        progress = (current_step / total_steps) * (end_point - start_point) + start_point

                        # Report progress
                        logger.info(f"Batch {batch_idx + 1}/{n_batches} - Loss: {loss:.4f} - Progress: {progress:.1f}%")

                        # Update metrics
                        await self.update_progress({
                            'progress': progress,
                            'current_batch': batch_idx + 1,
                            'total_batches': n_batches,
                            'current_loss': loss if loss is not None else 0.0,
                            'shard_id': shard_id,
                            'current_epoch': epoch + 1,
                            'total_epochs': self.config.epochs
                        })

                        await asyncio.sleep(0.1)

                    except Exception as e:
                        logger.error(f"Error in batch {batch_idx}: {str(e)}")
                        continue

                # Epoch completed
                avg_loss = epoch_loss / max(batch_count, 1)  # Prevent division by zero
                accuracy = min(0.99, 1.0 - avg_loss) if avg_loss > 0 else 0.0

                self.training_metrics["loss"].append(avg_loss)
                self.training_metrics["accuracy"].append(accuracy)

                logger.info(f"Epoch {epoch + 1} completed - Average Loss: {avg_loss:.4f}")

                # Save best model
                if avg_loss < best_loss and batch_count > 0:
                    best_loss = avg_loss
                    checkpoint_path = self.checkpoint_manager.save_checkpoint(
                        self.model,
                        self.optimizer,
                        epoch,
                        {
                            "loss": avg_loss,
                            "shard_info": shard_info
                        },
                        f"{task_id}_shard_{shard_id}"
                    )
                    logger.info(f"New best model saved: {checkpoint_path}")

                # Report epoch metrics
                await self.update_progress({
                    'epoch': epoch + 1,
                    'loss': avg_loss,
                    'accuracy': accuracy,
                    'progress': progress,
                    'shard_id': shard_id
                })

            # Training completed
            logger.info(f"\nTraining completed for shard {shard_id}!")
            self.training_metrics["status"] = "completed"
            return {
                'status': 'completed',
                'metrics': self.training_metrics,
                'best_loss': best_loss,
                'shard_id': shard_id
            }

        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}\n{traceback.format_exc()}")
            self.training_metrics["status"] = "failed"
            self.training_metrics["error"] = str(e)
            return {
                'status': 'failed',
                'error': str(e),
                'metrics': self.training_metrics,
                'shard_id': shard_info.get('id', 0)
            }

    def setup_distributed(self, world_size: int, rank: int):
        if not self.config.distributed:
            return

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                init_method='tcp://localhost:23456',
                world_size=world_size,
                rank=rank
            )

    def cleanup_distributed(self):
        if self.config.distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    def report_progress(self, task_id: str, metrics: Dict):
        logger.info(f"Task {task_id} progress: {json.dumps(metrics)}")

    def report_best_model(self, task_id: str, checkpoint_path: str, metrics: Dict):
        logger.info(f"Task {task_id} new best model: {checkpoint_path} metrics: {json.dumps(metrics)}")