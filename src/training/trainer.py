import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # For logging
import os
import time
import logging
from typing import Dict, Any, Union, List, Tuple # Added Tuple
from abc import ABC, abstractmethod # For abstract base class

from .losses import NoisePredictionLoss, MechanicalPropertyLoss, GeneratedCurveMatchingLoss
from .optimizers import get_optimizer, get_lr_scheduler
from ..models.diffusion_model import SpacedDiffusion

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BaseTrainer(ABC):
    """
    Abstract Base Class for training generative models.
    """
    def __init__(self,
                 model: nn.Module, # Can be a single model or a dict of models (e.g., for GANs)
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer_config: Union[Dict[str, Any], List[Dict[str, Any]]], # Single or list for GANs
                 scheduler_config: Union[Dict[str, Any], List[Dict[str, Any]], None], # Single, list, or None
                 loss_config: Dict[str, Any],
                 epochs: int,
                 device: Union[str, torch.device],
                 log_dir: str = 'runs/',
                 checkpoint_dir: str = 'checkpoints/',
                 save_interval: int = 10,
                 eval_interval: int = 1
                ):
        self.device = torch.device(device)
        self.model = model # This might be a single nn.Module or a dict of them
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)
        elif isinstance(self.model, dict): # For GANs
            for m_name, m_instance in self.model.items():
                self.model[m_name] = m_instance.to(self.device)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.start_epoch = 1 # For resuming training
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.loss_config = loss_config
        self.best_val_loss = float('inf')


        # --- Setup Optimizer(s) and LR Scheduler(s) ---
        self._setup_optimizers_and_schedulers(optimizer_config, scheduler_config)

        # --- Setup Logging and Checkpointing ---
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, self.__class__.__name__)) # Subdirectory for each trainer type

        logging.info(f"{self.__class__.__name__} initialized.")

    def _setup_optimizers_and_schedulers(self, optimizer_config, scheduler_config):
        """Helper to set up single or multiple optimizers/schedulers."""
        if isinstance(optimizer_config, list): # For GANs with multiple optimizers
            self.optimizer = {}
            self.scheduler = {} if scheduler_config else None
            model_names = list(self.model.keys()) # Assuming model is a dict for GANs

            if not isinstance(scheduler_config, list) and scheduler_config is not None:
                logging.warning("Optimizer config is a list, but scheduler config is not. Schedulers might not match optimizers.")

            for i, opt_conf in enumerate(optimizer_config):
                model_key = opt_conf.get('model_name', model_names[i]) # Assign optimizer to model
                if model_key not in self.model:
                    raise ValueError(f"Model name '{model_key}' in optimizer config not found in self.model.")

                self.optimizer[model_key] = get_optimizer(
                    self.model[model_key],
                    optimizer_name=opt_conf['name'],
                    learning_rate=opt_conf['lr'],
                    weight_decay=opt_conf.get('weight_decay', 0.0)
                )
                if self.scheduler and scheduler_config and i < len(scheduler_config):
                    sch_conf = scheduler_config[i]
                    self.scheduler[model_key] = get_lr_scheduler(
                        self.optimizer[model_key],
                        scheduler_name=sch_conf['name'],
                        **sch_conf.get('params', {})
                    )
                elif self.scheduler:
                     self.scheduler[model_key] = None # No scheduler for this optimizer
            logging.info("Multiple optimizers and schedulers configured.")

        else: # Single optimizer
            self.optimizer = get_optimizer(
                self.model, # Assumes self.model is a single nn.Module
                optimizer_name=optimizer_config['name'],
                learning_rate=optimizer_config['lr'],
                weight_decay=optimizer_config.get('weight_decay', 0.0)
            )
            if scheduler_config and scheduler_config['name'] != 'none':
                self.scheduler = get_lr_scheduler(
                    self.optimizer,
                    scheduler_name=scheduler_config['name'],
                    **scheduler_config.get('params', {})
                )
            else:
                self.scheduler = None
            logging.info("Single optimizer and scheduler configured.")

    @abstractmethod
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Runs a single training epoch. Must be implemented by subclasses.
        Should return a dictionary of training losses.
        """
        pass

    @abstractmethod
    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Evaluates the model on the validation set. Must be implemented by subclasses.
        Should return a dictionary of validation losses/metrics.
        """
        pass

    def train(self):
        """
        Starts the main training loop.
        """
        logging.info(f"Starting training with {self.__class__.__name__}...")

        for epoch in range(self.start_epoch, self.epochs + 1):
            train_losses = self._train_epoch(epoch)
            self._log_metrics(train_losses, epoch, 'Train')

            # Step learning rate scheduler(s) if not ReduceLROnPlateau
            self._step_schedulers(epoch, val_metric=None, is_plateau_type=False)


            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                val_metrics = self._validate_epoch(epoch)
                self._log_metrics(val_metrics, epoch, 'Validation')

                # For ReduceLROnPlateau, step based on a validation metric
                # Assuming the first metric returned by _validate_epoch is the one to monitor
                main_val_metric_name = list(val_metrics.keys())[0] if val_metrics else None
                main_val_metric_value = val_metrics.get(main_val_metric_name) if main_val_metric_name else None

                self._step_schedulers(epoch, val_metric=main_val_metric_value, is_plateau_type=True)

                # Save best model based on a primary validation loss/metric
                if main_val_metric_value is not None and main_val_metric_value < self.best_val_loss:
                    self.best_val_loss = main_val_metric_value
                    logging.info(f"Epoch {epoch}: New best validation metric ({main_val_metric_name}: {self.best_val_loss:.4f}). Saving model.")
                    self.save_checkpoint(epoch, is_best=True)


            if epoch % self.save_interval == 0 or epoch == self.epochs:
                self.save_checkpoint(epoch, is_best=False) # Regular save

        logging.info(f"Training finished for {self.__class__.__name__}.")
        self.writer.close()

    def _log_metrics(self, metrics: Dict[str, float], epoch: int, stage: str):
        """Logs metrics to console and TensorBoard."""
        log_message = f"{stage} Epoch [{epoch}/{self.epochs}]"
        for name, value in metrics.items():
            log_message += f", {name}: {value:.4f}"
            self.writer.add_scalar(f'{stage}_{self.__class__.__name__}/{name}', value, epoch)
        logging.info(log_message)
        if stage == 'Train' and self.scheduler: # Log LR for training
            if isinstance(self.optimizer, dict): # GAN case
                 for i, (opt_name, opt_instance) in enumerate(self.optimizer.items()):
                      self.writer.add_scalar(f'LearningRate/{opt_name}', opt_instance.param_groups[0]['lr'], epoch)
            else: # Single optimizer case
                 self.writer.add_scalar('LearningRate/main', self.optimizer.param_groups[0]['lr'], epoch)


    def _step_schedulers(self, epoch: int, val_metric: float = None, is_plateau_type: bool = False):
        """Steps single or multiple schedulers."""
        if not self.scheduler:
            return

        if isinstance(self.scheduler, dict): # GAN case
            for sch_name, sch_instance in self.scheduler.items():
                if sch_instance:
                    if isinstance(sch_instance, ReduceLROnPlateau):
                        if is_plateau_type and val_metric is not None:
                             sch_instance.step(val_metric)
                    elif not is_plateau_type:
                        sch_instance.step()
        else: # Single scheduler case
            if isinstance(self.scheduler, ReduceLROnPlateau):
                if is_plateau_type and val_metric is not None:
                     self.scheduler.step(val_metric)
            elif not is_plateau_type:
                self.scheduler.step()


    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves the model and optimizer state."""
        checkpoint_name = f'checkpoint_epoch_{epoch}.pt'
        if is_best:
            checkpoint_name = 'best_model.pt'
        checkpoint_path = os.path.join(self.checkpoint_dir, self.__class__.__name__, checkpoint_name)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)


        state = {'epoch': epoch, 'best_val_loss': self.best_val_loss}

        if isinstance(self.model, nn.Module):
            state['model_state_dict'] = self.model.state_dict()
            state['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler:
                state['scheduler_state_dict'] = self.scheduler.state_dict()
        elif isinstance(self.model, dict): # GANs
            state['model_state_dict'] = {name: m.state_dict() for name, m in self.model.items()}
            state['optimizer_state_dict'] = {name: o.state_dict() for name, o in self.optimizer.items()}
            if self.scheduler:
                state['scheduler_state_dict'] = {name: s.state_dict() for name, s in self.scheduler.items() if s}

        # Add model/training config for reproducibility
        state['config'] = {
            'loss_config': self.loss_config,
            # Add relevant parts of model_cfg, train_cfg if needed
        }

        try:
            torch.save(state, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error saving checkpoint {checkpoint_path}: {e}")

    def load_checkpoint(self, checkpoint_path: str):
        """Loads the model and optimizer state from a checkpoint."""
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if isinstance(self.model, nn.Module):
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            elif isinstance(self.model, dict): # GANs
                for name, model_state in checkpoint['model_state_dict'].items():
                    self.model[name].load_state_dict(model_state)
                for name, opt_state in checkpoint['optimizer_state_dict'].items():
                    self.optimizer[name].load_state_dict(opt_state)
                if self.scheduler and 'scheduler_state_dict' in checkpoint:
                    for name, sch_state in checkpoint['scheduler_state_dict'].items():
                        if self.scheduler.get(name) and sch_state:
                             self.scheduler[name].load_state_dict(sch_state)


            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

            logging.info(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {self.start_epoch}.")
            return self.start_epoch
        except Exception as e:
            logging.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            raise


class DiffusionModelTrainer(BaseTrainer):
    """
    Trainer for ConditionalDiffusionModel.
    """
    def __init__(self,
                 model: nn.Module, # Specific model type
                 diffusion: SpacedDiffusion,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer_config: Dict[str, Any],
                 scheduler_config: Dict[str, Any],
                 loss_config: Dict[str, Any],
                 epochs: int,
                 device: Union[str, torch.device],
                 log_dir: str = 'runs/',
                 checkpoint_dir: str = 'checkpoints/',
                 save_interval: int = 10,
                 eval_interval: int = 1):
        super().__init__(model, train_dataloader, val_dataloader, optimizer_config, scheduler_config,
                         loss_config, epochs, device, log_dir, checkpoint_dir, save_interval, eval_interval)

        # --- Setup Diffusion-Specific Loss Functions ---
        self.diffusion = diffusion
        self.mech_prop_loss_fn = None
        self.curve_match_loss_fn = None

        self.noise_loss_weight = self.loss_config.get('noise_weight', 1.0)
        self.mech_prop_loss_weight = self.loss_config.get('mech_prop_weight', 0.0)
        self.curve_match_loss_weight = self.loss_config.get('curve_match_weight', 0.0)

        if self.mech_prop_loss_weight > 0:
            self.mech_prop_loss_fn = MechanicalPropertyLoss(self.loss_config.get('mech_prop_weights'))
            logging.info("Mechanical property loss enabled for DiffusionModelTrainer.")

        if self.curve_match_loss_weight > 0:
            self.curve_match_loss_fn = GeneratedCurveMatchingLoss(self.loss_config.get('curve_match_type', 'mse'))
            logging.info("Generated curve matching loss enabled for DiffusionModelTrainer.")


    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_losses = {'total_loss': 0.0, 'noise_loss': 0.0, 'mech_prop_loss': 0.0, 'curve_match_loss': 0.0}
        num_batches = len(self.train_dataloader)

        start_time = time.time()
        for batch_idx, batch in enumerate(self.train_dataloader):
            x_0 = batch['fe_curve'].to(self.device)
            sequence_data = batch['sequence_data']
            if isinstance(sequence_data, torch.Tensor):
                sequence_data = sequence_data.to(self.device)
            conditions = batch['conditions'].to(self.device)
            t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],), device=self.device)
            model_kwargs = dict(y=conditions)

            self.optimizer.zero_grad()

            # Diffusion forward returns training loss
            current_total_loss = 0.0

            # 1. Noise Prediction Loss
            loss_dict = self.diffusion.training_losses(self.model, x_0, t, model_kwargs)
            noise_loss = loss_dict["loss"].mean()
            current_total_loss += self.noise_loss_weight * noise_loss
            running_losses['noise_loss'] += noise_loss.item()

            """
                        # 2. Optional Mechanical Property Loss
            if self.mech_prop_loss_fn and self.mech_prop_loss_weight > 0:
                # This requires predicting x_0 from (x_t, predicted_noise, t)
                # which might need a helper method in the ConditionalDiffusionModel
                # For now, this part is illustrative.
                try:
                    # Example: self.model._predict_x0_from_noise(x_t, t, predicted_noise)
                    # For simplicity, let's assume a placeholder or skip if not implemented
                    predicted_x0_for_loss = self.model._predict_x0_from_noise(x_t.detach(), t.detach(), predicted_noise) # Detach if needed
                    mech_loss = self.mech_prop_loss_fn(predicted_x0_for_loss, x_0)
                    current_total_loss += self.mech_prop_loss_weight * mech_loss
                    running_losses['mech_prop_loss'] += mech_loss.item()
                except AttributeError: # if _predict_x0_from_noise not in model
                     logging.debug(f"Epoch {epoch}, Batch {batch_idx}: Skipping mech prop loss as _predict_x0_from_noise not found in model.")
                except Exception as e:
                    logging.warning(f"Epoch {epoch}, Batch {batch_idx}: Error in mech prop loss calculation: {e}")


            # 3. Optional Generated Curve Matching Loss
            if self.curve_match_loss_fn and self.curve_match_loss_weight > 0:
                try:
                    predicted_x0_for_loss = self.model._predict_x0_from_noise(x_t.detach(), t.detach(), predicted_noise) # Detach if needed
                    curve_match_loss = self.curve_match_loss_fn(predicted_x0_for_loss, x_0)
                    current_total_loss += self.curve_match_loss_weight * curve_match_loss
                    running_losses['curve_match_loss'] += curve_match_loss.item()
                except AttributeError:
                     logging.debug(f"Epoch {epoch}, Batch {batch_idx}: Skipping curve match loss as _predict_x0_from_noise not found in model.")
                except Exception as e:
                    logging.warning(f"Epoch {epoch}, Batch {batch_idx}: Error in curve match loss calculation: {e}")
            """

            current_total_loss.backward()
            self.optimizer.step()
            running_losses['total_loss'] += current_total_loss.item()

        epoch_losses = {name: (loss_sum / num_batches if num_batches > 0 else 0) for name, loss_sum in running_losses.items()}
        epoch_losses['time_seconds'] = time.time() - start_time
        return epoch_losses

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_losses = {'total_loss': 0.0, 'noise_loss': 0.0, 'mech_prop_loss': 0.0, 'curve_match_loss': 0.0}
        num_batches = len(self.val_dataloader)
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                x_0 = batch['fe_curve'].to(self.device)
                sequence_data = batch['sequence_data']
                if isinstance(sequence_data, torch.Tensor):
                    sequence_data = sequence_data.to(self.device)
                conditions = batch['conditions'].to(self.device)
                t = torch.randint(0, self.diffusion.num_timesteps, (x_0.shape[0],), device=self.device)
                model_kwargs = dict(y=conditions)

                self.optimizer.zero_grad()

                # Diffusion forward returns training loss
                current_total_loss = 0.0

                # 1. Noise Prediction Loss
                loss_dict = self.diffusion.training_losses(self.model, x_0, t, model_kwargs)
                noise_loss = loss_dict["loss"].mean()
                current_total_loss += self.noise_loss_weight * noise_loss
                running_losses['noise_loss'] += noise_loss.item()

                if self.mech_prop_loss_fn and self.mech_prop_loss_weight > 0:
                    try:
                        predicted_x0_for_loss = self.model._predict_x0_from_noise(x_t, t, predicted_noise)
                        mech_loss = self.mech_prop_loss_fn(predicted_x0_for_loss, x_0)
                        current_total_loss += self.mech_prop_loss_weight * mech_loss
                        running_losses['mech_prop_loss'] += mech_loss.item()
                    except AttributeError: pass
                    except Exception: pass


                if self.curve_match_loss_fn and self.curve_match_loss_weight > 0:
                    try:
                        predicted_x0_for_loss = self.model._predict_x0_from_noise(x_t, t, predicted_noise)
                        curve_match_loss = self.curve_match_loss_fn(predicted_x0_for_loss, x_0)
                        current_total_loss += self.curve_match_loss_weight * curve_match_loss
                        running_losses['curve_match_loss'] += curve_match_loss.item()
                    except AttributeError: pass
                    except Exception: pass

                running_losses['total_loss'] += current_total_loss.item()

        epoch_losses = {name: (loss_sum / num_batches if num_batches > 0 else 0) for name, loss_sum in running_losses.items()}
        epoch_losses['time_seconds'] = time.time() - start_time
        return epoch_losses


class VAETrainer(BaseTrainer):
    """
    Trainer for Variational Autoencoder (VAE) models.
    """
    def __init__(self,
                 model: nn.Module, # Should be your VAEModel
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer_config: Dict[str, Any],
                 scheduler_config: Dict[str, Any],
                 loss_config: Dict[str, Any], # e.g., {'reconstruction_weight': 1.0, 'kl_weight': 0.001}
                 epochs: int,
                 device: Union[str, torch.device],
                 log_dir: str = 'runs/',
                 checkpoint_dir: str = 'checkpoints/',
                 save_interval: int = 10,
                 eval_interval: int = 1):
        super().__init__(model, train_dataloader, val_dataloader, optimizer_config, scheduler_config,
                         loss_config, epochs, device, log_dir, checkpoint_dir, save_interval, eval_interval)

        # --- VAE Specific Loss ---
        # self.vae_loss_fn = VAELoss(reconstruction_weight=self.loss_config.get('reconstruction_weight', 1.0),
        #                            kl_weight=self.loss_config.get('kl_weight', 1.0))
        # For now, let's assume the VAE model's forward pass returns reconstruction_loss and kl_loss directly
        logging.info("VAETrainer initialized. Ensure VAE model forward returns 'reconstruction_loss' and 'kl_loss'.")


    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        running_losses = {'total_loss': 0.0, 'reconstruction_loss': 0.0, 'kl_loss': 0.0}
        num_batches = len(self.train_dataloader)
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_dataloader):
            # VAEs typically take x_0 as input and try to reconstruct it
            x_0 = batch['fe_curve'].to(self.device)
            # VAEs can also be conditional, so handle sequence_data and conditions
            sequence_data = batch.get('sequence_data').to(self.device) # .get() for optional
            conditions = batch.get('conditions').to(self.device)

            self.optimizer.zero_grad()

            # VAE model forward should return: reconstructed_x, mu, logvar
            # Loss calculation is often done inside the model or a separate VAELoss class
            # For simplicity, assume model.forward returns a dict of losses or individual losses
            # Example: output_dict = self.model(x_0, sequence_data, conditions)
            # recon_loss = output_dict['reconstruction_loss']
            # kl_div = output_dict['kl_loss']

            # --- Placeholder for VAE forward and loss ---
            # This needs to be adapted to your specific VAE model implementation
            # Assuming model forward returns (reconstructed_x, mu, logvar)
            try:
                reconstructed_x, mu, logvar = self.model(x_0, conditions) # Adapt to your VAE input
                recon_loss = nn.functional.mse_loss(reconstructed_x, x_0, reduction='sum') / x_0.size(0) # Example MSE per sample
                # KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_0.size(0) # Per sample
            except Exception as e:
                 logging.error(f"Error in VAE forward/loss: {e}. Using dummy losses.")
                 recon_loss = torch.tensor(0.0, device=self.device)
                 kl_div = torch.tensor(0.0, device=self.device)
            # --------------------------------------------


            reconstruction_weight = self.loss_config.get('reconstruction_weight', 1.0)
            kl_weight = self.loss_config.get('kl_weight', 1.0) # Beta-VAE factor

            current_total_loss = reconstruction_weight * recon_loss + kl_weight * kl_div

            current_total_loss.backward()
            self.optimizer.step()

            running_losses['total_loss'] += current_total_loss.item()
            running_losses['reconstruction_loss'] += recon_loss.item()
            running_losses['kl_loss'] += kl_div.item()

        epoch_losses = {name: (loss_sum / num_batches if num_batches > 0 else 0) for name, loss_sum in running_losses.items()}
        epoch_losses['time_seconds'] = time.time() - start_time
        return epoch_losses



    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        running_losses = {'total_loss': 0.0, 'reconstruction_loss': 0.0, 'kl_loss': 0.0}
        num_batches = len(self.val_dataloader)
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                x_0 = batch['fe_curve'].to(self.device)
                sequence_data = batch.get('sequence_data')
                conditions = batch.get('conditions')
                if sequence_data is not None and isinstance(sequence_data, torch.Tensor):
                    sequence_data = sequence_data.to(self.device)
                if conditions is not None and isinstance(conditions, torch.Tensor):
                    conditions = conditions.to(self.device)

                # --- Placeholder for VAE forward and loss ---
                try:
                    reconstructed_x, mu, logvar = self.model(x_0, conditions)
                    recon_loss = nn.functional.mse_loss(reconstructed_x, x_0, reduction='sum') / x_0.size(0)
                    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x_0.size(0)
                except Exception:
                    recon_loss = torch.tensor(0.0, device=self.device)
                    kl_div = torch.tensor(0.0, device=self.device)
                # --------------------------------------------

                reconstruction_weight = self.loss_config.get('reconstruction_weight', 1.0)
                kl_weight = self.loss_config.get('kl_weight', 1.0)
                current_total_loss = reconstruction_weight * recon_loss + kl_weight * kl_div

                running_losses['total_loss'] += current_total_loss.item()
                running_losses['reconstruction_loss'] += recon_loss.item()
                running_losses['kl_loss'] += kl_div.item()

        epoch_losses = {name: (loss_sum / num_batches if num_batches > 0 else 0) for name, loss_sum in running_losses.items()}
        epoch_losses['time_seconds'] = time.time() - start_time
        return epoch_losses


class GANTrainer(BaseTrainer):
    """
    Trainer for Generative Adversarial Network (GAN) models.
    Assumes model is a dict: {'generator': GenModel, 'discriminator': DiscModel}
    Assumes optimizer_config and scheduler_config are lists of two dicts,
    one for generator and one for discriminator.
    """
    def __init__(self,
                 models: Dict[str, nn.Module], # {'generator': Gen, 'discriminator': Disc}
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader, # Validation for GANs can be tricky (e.g., FID score)
                 optimizer_configs: List[Dict[str, Any]], # [{'model_name':'generator', ...}, {'model_name':'discriminator', ...}]
                 scheduler_configs: List[Dict[str, Any]],
                 loss_config: Dict[str, Any], # e.g., weights for G/D losses, feature matching
                 epochs: int,
                 device: Union[str, torch.device],
                 log_dir: str = 'runs/',
                 checkpoint_dir: str = 'checkpoints/',
                 save_interval: int = 10,
                 eval_interval: int = 1,
                 k_critic_steps: int = 1 # For WGAN: number of critic updates per generator update
                ):
        super().__init__(models, train_dataloader, val_dataloader, optimizer_configs, scheduler_configs,
                         loss_config, epochs, device, log_dir, checkpoint_dir, save_interval, eval_interval)

        self.generator = self.model['generator']
        self.discriminator = self.model['discriminator']
        self.optimizer_g = self.optimizer['generator']
        self.optimizer_d = self.optimizer['discriminator']
        self.scheduler_g = self.scheduler.get('generator') if self.scheduler else None
        self.scheduler_d = self.scheduler.get('discriminator') if self.scheduler else None
        self.k_critic_steps = k_critic_steps


        # --- GAN Specific Losses ---
        # self.criterion_g = GANGeneratorLoss(...)
        # self.criterion_d = GANDiscriminatorLoss(...)
        # For simplicity, assume binary cross entropy for now
        self.adversarial_loss = nn.BCEWithLogitsLoss() # Or MSELoss for LSGAN
        # Or use specific loss functions from your loss_config
        logging.info("GANTrainer initialized. Ensure GAN loss functions are set up.")

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        running_losses = {'g_loss': 0.0, 'd_loss_real': 0.0, 'd_loss_fake':0.0, 'd_loss_total': 0.0}
        num_batches = len(self.train_dataloader)
        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_dataloader):
            real_samples = batch['fe_curve'].to(self.device) # Assuming F-E curves are real samples
            batch_size = real_samples.size(0)

            # Latent vector for generator input
            # Noise dimension should be part of generator's config
            z_dim = self.generator.z_dim if hasattr(self.generator, 'z_dim') else 100 # Placeholder
            z = torch.randn(batch_size, z_dim, device=self.device)

            # Conditional GAN: Get conditions
            sequence_data = batch.get('sequence_data')
            conditions_input = batch.get('conditions')
            if sequence_data is not None and isinstance(sequence_data, torch.Tensor):
                sequence_data = sequence_data.to(self.device)
            if conditions_input is not None and isinstance(conditions_input, torch.Tensor):
                conditions_input = conditions_input.to(self.device)
            # Concatenate or process sequence_data and conditions_input into a single conditional vector if needed by G and D

            # --- Train Discriminator ---
            for _ in range(self.k_critic_steps): # WGAN-like update schedule
                self.optimizer_d.zero_grad()

                # Real samples
                d_real_output = self.discriminator(real_samples, sequence_data, conditions_input) # Adapt to your D input
                d_loss_real = self.adversarial_loss(d_real_output, torch.ones_like(d_real_output))

                # Fake samples
                fake_samples = self.generator(z, sequence_data, conditions_input).detach() # Detach to avoid training G here
                d_fake_output = self.discriminator(fake_samples, sequence_data, conditions_input)
                d_loss_fake = self.adversarial_loss(d_fake_output, torch.zeros_like(d_fake_output))

                # Total discriminator loss
                d_loss_total = (d_loss_real + d_loss_fake) / 2
                d_loss_total.backward()
                self.optimizer_d.step()

                # Optional: WGAN-GP gradient penalty (not implemented here)

            running_losses['d_loss_real'] += d_loss_real.item()
            running_losses['d_loss_fake'] += d_loss_fake.item()
            running_losses['d_loss_total'] += d_loss_total.item()


            # --- Train Generator ---
            self.optimizer_g.zero_grad()
            # Generate new fake samples (no detach this time)
            fake_samples_for_g = self.generator(z, sequence_data, conditions_input)
            g_fake_output = self.discriminator(fake_samples_for_g, sequence_data, conditions_input)
            g_loss = self.adversarial_loss(g_fake_output, torch.ones_like(g_fake_output)) # Try to fool D

            # Optional: Add other generator losses (e.g., feature matching, perceptual loss)
            # g_loss += some_other_g_loss_weight * calculate_other_g_loss(...)

            g_loss.backward()
            self.optimizer_g.step()
            running_losses['g_loss'] += g_loss.item()

        epoch_losses = {name: (loss_sum / num_batches if num_batches > 0 else 0) for name, loss_sum in running_losses.items()}
        epoch_losses['time_seconds'] = time.time() - start_time
        return epoch_losses

    def _validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.generator.eval()
        self.discriminator.eval()
        # Validation for GANs is often qualitative (visual inspection of samples)
        # or uses metrics like Inception Score, FID (Frechet Inception Distance).
        # These are complex to implement here and depend on the data modality.
        # For SMFS curves, you might compare distributions of generated vs. real mechanical properties.

        # Placeholder: Generate some samples and calculate a dummy metric
        # Or simply log generator/discriminator loss on val set if meaningful
        running_losses = {'val_g_loss': 0.0, 'val_d_loss': 0.0} # Example
        num_batches = len(self.val_dataloader)
        start_time = time.time()

        if num_batches == 0:
            return {'val_g_loss': 0.0, 'val_d_loss': 0.0, 'time_seconds': 0.0} # No validation data

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                real_samples = batch['fe_curve'].to(self.device)
                batch_size = real_samples.size(0)
                z_dim = self.generator.z_dim if hasattr(self.generator, 'z_dim') else 100
                z = torch.randn(batch_size, z_dim, device=self.device)

                sequence_data = batch.get('sequence_data')
                conditions_input = batch.get('conditions')
                if sequence_data is not None and isinstance(sequence_data, torch.Tensor): sequence_data = sequence_data.to(self.device)
                if conditions_input is not None and isinstance(conditions_input, torch.Tensor): conditions_input = conditions_input.to(self.device)


                # Discriminator validation loss
                d_real_output = self.discriminator(real_samples, sequence_data, conditions_input)
                d_loss_real_val = self.adversarial_loss(d_real_output, torch.ones_like(d_real_output))
                fake_samples_val = self.generator(z, sequence_data, conditions_input)
                d_fake_output_val = self.discriminator(fake_samples_val, sequence_data, conditions_input)
                d_loss_fake_val = self.adversarial_loss(d_fake_output_val, torch.zeros_like(d_fake_output_val))
                d_loss_total_val = (d_loss_real_val + d_loss_fake_val) / 2
                running_losses['val_d_loss'] += d_loss_total_val.item()

                # Generator validation loss
                g_fake_output_val = self.discriminator(fake_samples_val, sequence_data, conditions_input) # Re-use fake samples
                g_loss_val = self.adversarial_loss(g_fake_output_val, torch.ones_like(g_fake_output_val))
                running_losses['val_g_loss'] += g_loss_val.item()

        epoch_losses = {name: (loss_sum / num_batches if num_batches > 0 else 0) for name, loss_sum in running_losses.items()}
        epoch_losses['time_seconds'] = time.time() - start_time

        # Log some generated samples (if applicable)
        # self.log_generated_samples(epoch, num_samples_to_log=5)
        return epoch_losses

    # def log_generated_samples(self, epoch, num_samples_to_log=5):
    #     """Helper to log some generated samples during validation (e.g., to TensorBoard)."""
    #     self.generator.eval()
    #     with torch.no_grad():
    #         z_dim = self.generator.z_dim if hasattr(self.generator, 'z_dim') else 100
    #         fixed_z = torch.randn(num_samples_to_log, z_dim, device=self.device)
    #         # Need fixed conditional inputs too if conditional GAN
    #         # fixed_conditions = ...
    #         generated_samples = self.generator(fixed_z, fixed_conditions)
    #         # Assuming F-E curves can be plotted (e.g., using matplotlib and add_figure to TensorBoard)
    #         # This part is highly specific to your data and visualization needs.
    #         # For example, plot curves and add as images:
    #         # fig, axes = plt.subplots(1, num_samples_to_log, figsize=(15, 3))
    #         # for i in range(num_samples_to_log):
    #         #     axes[i].plot(generated_samples[i].cpu().numpy().squeeze()) # Assuming 1D curve
    #         # self.writer.add_figure(f'Generated_Samples_Epoch_{epoch}', fig, global_step=epoch)
    #     self.generator.train()
