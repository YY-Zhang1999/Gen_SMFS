# src/training/trainer.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # For logging
import os
import time
import json
import logging
from typing import Dict, Any, Union, List

# Import necessary modules from your project
try:
    from models.diffusion_model import ConditionalDiffusionModel
    from training.losses import NoisePredictionLoss, MechanicalPropertyLoss, GeneratedCurveMatchingLoss
    from training.optimizers import get_optimizer, get_lr_scheduler
    from data_processing.dataset import FEDataset # For type hinting
    from data_processing.preprocessing import encode_protein_sequences # Needed if dataset returns raw sequences
    # Import other necessary components/configs
    # from config.model_config import ModelConfig # Example config loading
    # from config.training_config import TrainingConfig
except ImportError as e:
    logging.error(f"Failed to import required modules: {e}")
    logging.error("Please ensure src directory is in your PYTHONPATH or run from the project root.")
    # Exit or raise error if imports fail
    raise

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Trainer:
    """
    Manages the training and evaluation loop for the ConditionalDiffusionModel.
    """
    def __init__(self,
                 model: ConditionalDiffusionModel,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 optimizer_config: Dict[str, Any],
                 scheduler_config: Dict[str, Any],
                 loss_config: Dict[str, Any],
                 epochs: int,
                 device: Union[str, torch.device],
                 log_dir: str = 'runs/',
                 checkpoint_dir: str = 'checkpoints/',
                 save_interval: int = 10, # Save checkpoint every N epochs
                 eval_interval: int = 5 # Evaluate on validation set every N epochs
                ):
        """
        Initializes the Trainer.

        Args:
            model (ConditionalDiffusionModel): The Diffusion Model to train.
            train_dataloader (DataLoader): DataLoader for the training dataset.
            val_dataloader (DataLoader): DataLoader for the validation dataset.
            optimizer_config (Dict[str, Any]): Dictionary containing optimizer configuration
                                                (e.g., {'name': 'adam', 'lr': 1e-4, 'weight_decay': 1e-5}).
            scheduler_config (Dict[str, Any]): Dictionary containing LR scheduler configuration
                                                (e.g., {'name': 'step', 'step_size': 30, 'gamma': 0.1} or {'name': 'none'}).
            loss_config (Dict[str, Any]): Dictionary containing loss function configuration
                                          (e.g., {'noise_weight': 1.0, 'mech_prop_weight': 0.1, 'curve_match_weight': 0.1, 'mech_prop_weights': {'max_force': 0.5}}).
            epochs (int): Total number of epochs to train for.
            device (Union[str, torch.device]): The device to train on ('cuda' or 'cpu').
            log_dir (str): Directory for TensorBoard logs.
            checkpoint_dir (str): Directory to save model checkpoints.
            save_interval (int): Save checkpoint every this many epochs.
            eval_interval (int): Evaluate on validation set every this many epochs.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.eval_interval = eval_interval

        # --- Setup Optimizer ---
        self.optimizer = get_optimizer(
            self.model,
            optimizer_name=optimizer_config['name'],
            learning_rate=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 0.0) # Use .get for optional args
        )

        # --- Setup LR Scheduler ---
        self.scheduler = get_lr_scheduler(
            self.optimizer,
            scheduler_name=scheduler_config['name'],
            **scheduler_config.get('params', {}) # Pass scheduler-specific params
        )

        # --- Setup Loss Functions ---
        self.noise_loss_fn = NoisePredictionLoss()
        self.mech_prop_loss_fn = None
        self.curve_match_loss_fn = None

        self.noise_loss_weight = loss_config.get('noise_weight', 1.0)
        self.mech_prop_loss_weight = loss_config.get('mech_prop_weight', 0.0)
        self.curve_match_loss_weight = loss_config.get('curve_match_weight', 0.0)

        if self.mech_prop_loss_weight > 0:
            self.mech_prop_loss_fn = MechanicalPropertyLoss(loss_config.get('mech_prop_weights'))
            logging.warning("Mechanical property loss is enabled. Ensure your property extraction is differentiable.")

        if self.curve_match_loss_weight > 0:
             self.curve_match_loss_fn = GeneratedCurveMatchingLoss(loss_config.get('curve_match_type', 'mse'))
             logging.warning("Generated curve matching loss is enabled. This loss is typically applied to predicted x_0.")


        # --- Setup Logging and Checkpointing ---
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        logging.info("Trainer initialized.")


    def _train_epoch(self, epoch: int):
        """
        Runs a single training epoch.
        """
        self.model.train() # Set model to training mode
        running_loss = 0.0
        noise_loss_sum = 0.0
        mech_prop_loss_sum = 0.0
        curve_match_loss_sum = 0.0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch data to the device
            x_0 = batch['fe_curve'].to(self.device)
            # Sequence data handling depends on encoding type
            sequence_data = batch['sequence_data']
            if isinstance(sequence_data, torch.Tensor):
                 sequence_data = sequence_data.to(self.device)
            # If raw strings, it remains a list, and the ProteinEncoder handles it

            conditions = batch['conditions'].to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # Forward pass
            # The model forward returns x_t, t, predicted_noise, true_noise
            x_t, t, predicted_noise, true_noise = self.model(x_0, sequence_data, conditions)

            # Calculate total loss
            total_loss = 0.0

            # 1. Noise Prediction Loss (Primary Loss)
            noise_loss = self.noise_loss_fn(predicted_noise, true_noise)
            total_loss += self.noise_loss_weight * noise_loss
            noise_loss_sum += noise_loss.item()

            # 2. Optional Mechanical Property Loss
            if self.mech_prop_loss_fn is not None and self.mech_prop_loss_weight > 0:
                 # To calculate mechanical property loss during training on a noisy step (x_t)
                 # you might need to predict x_0 from predicted_noise:
                 # x_0_hat = (x_t - self.model.sqrt_one_minus_alphas_bar[t.cpu()].view(-1, 1, 1).to(self.device) * predicted_noise) / self.model.sqrt_alphas_bar[t.cpu()].view(-1, 1, 1).to(self.device)
                 # Or calculate properties directly on the potentially noisy x_t if meaningful.
                 # For simplicity in this placeholder, let's assume we calculate it on a *derived* x_0_hat
                 # This requires a detached x_0_hat if you only want the loss to influence the prediction of noise,
                 # or non-detached if you want it to influence the network differently.
                 # A common approach is to apply this loss only to the *final generated samples* during evaluation.
                 # If applying during training, ensure differentiability.

                 # Placeholder: Calculate loss comparing predicted_noise to true_noise
                 # This is NOT the correct way to apply Mech Prop Loss during training typically
                 # Replace this with calculation based on predicted x_0 or similar if desired
                 mech_prop_loss = torch.tensor(0.0, device=self.device) # Replace with actual calculation
                 # For a more correct approach, you'd calculate a quantity related to x_0 from x_t and predicted_noise
                 # e.g., predicted_x0 = self.model._predict_x0_from_noise(x_t, t, predicted_noise) # Needs a helper in the model
                 # mech_prop_loss = self.mech_prop_loss_fn(predicted_x0, x_0)

                 total_loss += self.mech_prop_loss_weight * mech_prop_loss
                 mech_prop_loss_sum += mech_prop_loss.item()
                 logging.debug(f"Epoch {epoch}, Batch {batch_idx}: Mechanical property loss placeholder used.")


            # 3. Optional Generated Curve Matching Loss
            if self.curve_match_loss_fn is not None and self.curve_match_loss_weight > 0:
                 # Similar to mech prop loss, this requires predicting x_0_hat
                 curve_match_loss = torch.tensor(0.0, device=self.device) # Replace with actual calculation
                 # e.g., predicted_x0 = self.model._predict_x0_from_noise(x_t, t, predicted_noise) # Needs a helper in the model
                 # curve_match_loss = self.curve_match_loss_fn(predicted_x0, x_0)

                 total_loss += self.curve_match_loss_weight * curve_match_loss
                 curve_match_loss_sum += curve_match_loss.item()
                 logging.debug(f"Epoch {epoch}, Batch {batch_idx}: Curve matching loss placeholder used.")


            # Backward pass and optimization
            total_loss.backward()
            self.optimizer.step()

            running_loss += total_loss.item()

            # Log batch loss to TensorBoard (optional, can be noisy)
            # self.writer.add_scalar('Loss/train_batch', total_loss.item(), epoch * len(self.train_dataloader) + batch_idx)

        end_time = time.time()
        epoch_loss = running_loss / len(self.train_dataloader)
        epoch_noise_loss = noise_loss_sum / len(self.train_dataloader)
        epoch_mech_prop_loss = mech_prop_loss_sum / len(self.train_dataloader)
        epoch_curve_match_loss = curve_match_loss_sum / len(self.train_dataloader)


        logging.info(f"Epoch [{epoch}/{self.epochs}], Loss: {epoch_loss:.4f}, Time: {end_time - start_time:.2f}s")
        logging.info(f"  Noise Loss: {epoch_noise_loss:.4f}")
        if self.mech_prop_loss_fn is not None:
            logging.info(f"  Mech Prop Loss (Placeholder): {epoch_mech_prop_loss:.4f}")
        if self.curve_match_loss_fn is not None:
             logging.info(f"  Curve Match Loss (Placeholder): {epoch_curve_match_loss:.4f}")


        # Log epoch losses to TensorBoard
        self.writer.add_scalar('Loss/train_epoch', epoch_loss, epoch)
        self.writer.add_scalar('Loss/train_noise', epoch_noise_loss, epoch)
        if self.mech_prop_loss_fn is not None:
             self.writer.add_scalar('Loss/train_mech_prop', epoch_mech_prop_loss, epoch)
        if self.curve_match_loss_fn is not None:
             self.writer.add_scalar('Loss/train_curve_match', epoch_curve_match_loss, epoch)

        # Step the learning rate scheduler if it's not ReduceLROnPlateau
        if self.scheduler is not None and not isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step()
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)


    def _validate(self, epoch: int) -> float:
        """
        Evaluates the model on the validation set.

        Returns:
            float: The average validation loss.
        """
        self.model.eval() # Set model to evaluation mode
        running_loss = 0.0
        noise_loss_sum = 0.0
        mech_prop_loss_sum = 0.0
        curve_match_loss_sum = 0.0

        start_time = time.time()

        with torch.no_grad(): # Disable gradient calculation
            for batch_idx, batch in enumerate(self.val_dataloader):
                # Move batch data to the device
                x_0 = batch['fe_curve'].to(self.device)
                sequence_data = batch['sequence_data']
                if isinstance(sequence_data, torch.Tensor):
                     sequence_data = sequence_data.to(self.device)
                conditions = batch['conditions'].to(self.device)

                # Forward pass (sample a random timestep for validation loss)
                x_t, t, predicted_noise, true_noise = self.model(x_0, sequence_data, conditions)

                # Calculate validation loss
                total_loss = 0.0
                noise_loss = self.noise_loss_fn(predicted_noise, true_noise)
                total_loss += self.noise_loss_weight * noise_loss
                noise_loss_sum += noise_loss.item()

                # Optional Mechanical Property Loss (apply to derived x_0_hat)
                if self.mech_prop_loss_fn is not None and self.mech_prop_loss_weight > 0:
                     # Predict x_0_hat for evaluation metrics/losses if needed
                     # predicted_x0 = self.model._predict_x0_from_noise(x_t, t, predicted_noise) # Needs a helper in the model
                     # mech_prop_loss = self.mech_prop_loss_fn(predicted_x0, x_0) # Compare predicted x_0 to true x_0

                     mech_prop_loss = torch.tensor(0.0, device=self.device) # Placeholder

                     total_loss += self.mech_prop_loss_weight * mech_prop_loss
                     mech_prop_loss_sum += mech_prop_loss.item()
                     logging.debug(f"Epoch {epoch}, Validation Batch {batch_idx}: Mechanical property loss placeholder used.")


                # Optional Generated Curve Matching Loss (apply to derived x_0_hat)
                if self.curve_match_loss_fn is not None and self.curve_match_loss_weight > 0:
                     # predicted_x0 = self.model._predict_x0_from_noise(x_t, t, predicted_noise) # Needs a helper in the model
                     # curve_match_loss = self.curve_match_loss_fn(predicted_x0, x_0)

                     curve_match_loss = torch.tensor(0.0, device=self.device) # Placeholder

                     total_loss += self.curve_match_loss_weight * curve_match_loss
                     curve_match_loss_sum += curve_match_loss.item()
                     logging.debug(f"Epoch {epoch}, Validation Batch {batch_idx}: Curve matching loss placeholder used.")


                running_loss += total_loss.item()


        end_time = time.time()
        val_loss = running_loss / len(self.val_dataloader)
        val_noise_loss = noise_loss_sum / len(self.val_dataloader)
        val_mech_prop_loss = mech_prop_loss_sum / len(self.val_dataloader)
        val_curve_match_loss = curve_match_loss_sum / len(self.val_dataloader)


        logging.info(f"Validation Epoch [{epoch}/{self.epochs}], Loss: {val_loss:.4f}, Time: {end_time - start_time:.2f}s")
        logging.info(f"  Validation Noise Loss: {val_noise_loss:.4f}")
        if self.mech_prop_loss_fn is not None:
            logging.info(f"  Validation Mech Prop Loss (Placeholder): {val_mech_prop_loss:.4f}")
        if self.curve_match_loss_fn is not None:
             logging.info(f"  Validation Curve Match Loss (Placeholder): {val_curve_match_loss:.4f}")


        # Log validation losses to TensorBoard
        self.writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        self.writer.add_scalar('Loss/val_noise', val_noise_loss, epoch)
        if self.mech_prop_loss_fn is not None:
             self.writer.add_scalar('Loss/val_mech_prop', val_mech_prop_loss, epoch)
        if self.curve_match_loss_fn is not None:
             self.writer.add_scalar('Loss/val_curve_match', val_curve_match_loss, epoch)


        # Step the learning rate scheduler if it's ReduceLROnPlateau
        if self.scheduler is not None and isinstance(self.scheduler, ReduceLROnPlateau):
            # ReduceLROnPlateau typically steps based on a validation metric
            # Here we use the total validation loss
            self.scheduler.step(val_loss)
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], epoch)


        return val_loss

    def train(self):
        """
        Starts the main training loop.
        """
        logging.info("Starting training...")
        best_val_loss = float('inf')

        for epoch in range(1, self.epochs + 1):
            self._train_epoch(epoch)

            # Evaluate on validation set
            if epoch % self.eval_interval == 0 or epoch == self.epochs:
                val_loss = self._validate(epoch)

                # Save best model based on validation loss (optional)
                # if val_loss < best_val_loss:
                #     best_val_loss = val_loss
                #     logging.info(f"Epoch {epoch}: New best validation loss achieved. Saving model.")
                #     self.save_checkpoint(epoch, is_best=True)


            # Save checkpoint periodically
            if epoch % self.save_interval == 0 or epoch == self.epochs:
                 self.save_checkpoint(epoch, is_best=False)


        logging.info("Training finished.")
        self.writer.close()


    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        Saves the model and optimizer state.

        Args:
            epoch (int): The current epoch number.
            is_best (bool): Whether this is the best model so far.
        """
        checkpoint_name = f'checkpoint_epoch_{epoch}.pt'
        if is_best:
            checkpoint_name = 'best_model.pt'

        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': getattr(self, 'best_val_loss', None), # Save if tracking best loss
            'config': { # Save relevant configurations
                'optimizer': self.optimizer.state_dict()['param_groups'][0], # Save initial params
                'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
                # Add model config, loss config etc.
            }
        }

        try:
            torch.save(state, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logging.error(f"Error saving checkpoint {checkpoint_path}: {e}")


    def load_checkpoint(self, checkpoint_path: str):
        """
        Loads the model and optimizer state from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint file not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None and self.scheduler is not None:
            #      self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
            # Update best_val_loss if tracking
            if 'best_val_loss' in checkpoint and hasattr(self, 'best_val_loss'):
                 self.best_val_loss = checkpoint['best_val_loss']

            logging.info(f"Checkpoint loaded from {checkpoint_path}. Starting from epoch {start_epoch}.")
            return start_epoch

        except Exception as e:
            logging.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            raise


# Example Usage (requires dummy dataset and model)
if __name__ == "__main__":
    print("--- Testing trainer.py ---")

    # Create dummy dataset, dataloaders, and model for testing
    # These would normally come from your data_processing and models modules
    class DummyDataset:
        def __init__(self, num_samples=100, fe_len=200, seq_dim=100, cond_dim=2):
            self.num_samples = num_samples
            self.fe_len = fe_len
            self.seq_dim = seq_dim
            self.cond_dim = cond_dim

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            # Simulate returning pre-encoded data
            return {
                'fe_curve': torch.randn(self.fe_len, 1), # FE curve (force)
                'sequence_data': torch.randn(self.seq_dim), # Pre-encoded sequence
                'conditions': torch.randn(self.cond_dim) # Conditions
            }

    class DummyProteinEncoder(nn.Module):
         def __init__(self, input_dim, output_dim, encoding_type='pretrained_embeddings', **kwargs):
            super().__init__()
            self.output_dim = output_dim
            self.fc = nn.Linear(input_dim, output_dim)
            self.encoding_type = encoding_type # For forward pass logic
         def forward(self, x):
            # Simulate handling tensor input
            return self.fc(x)

    class DummyConditionEncoder(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, output_dim)
        def forward(self, x): return self.fc(x)

    class DummyTimeEmbedding(nn.Module):
        def __init__(self, embed_dim):
            super().__init__()
            self.fc = nn.Linear(embed_dim, embed_dim) # Simple linear for placeholder
        def forward(self, t):
            # Simulate time embedding output
            return self.fc(torch.randn(t.size(0), self.fc.in_features))

    class DummyDenoisingModel(nn.Module):
         def __init__(self, fe_curve_length, fe_curve_channels, protein_embed_dim, condition_embed_dim, time_embed_dim, model_channels):
             super().__init__()
             # Simulate output matching fe_curve shape
             self.output_proj = nn.Linear(model_channels, fe_curve_channels)
             self.dummy_linear = nn.Linear(fe_curve_length * model_channels + protein_embed_dim + condition_embed_dim + time_embed_dim, fe_curve_length * model_channels) # Very rough placeholder to use inputs

             # Need to account for how conditional embeddings are used... simpler dummy:
             self.dummy_linear_alt = nn.Linear(fe_curve_length + protein_embed_dim + condition_embed_dim + time_embed_dim, fe_curve_length) # Sum dimensions for placeholder

             # Correcting to match how ConditionalDenoisingModel forward is called:
             # It expects x_t, time_emb, protein_emb, condition_emb
             # x_t shape: (B, L, C_fe), others are (B, E)
             # Output shape: (B, L, C_fe)
             # Let's make a dummy that takes concatenated flattened inputs and reshapes output
             total_flat_input_dim = fe_curve_length * fe_curve_channels + protein_embed_dim + condition_embed_dim + time_embed_dim
             self.fc1 = nn.Linear(total_flat_input_dim, fe_curve_length * fe_curve_channels * 2) # Expand
             self.fc2 = nn.Linear(fe_curve_length * fe_curve_channels * 2, fe_curve_length * fe_curve_channels) # Project back
             self.fe_curve_length = fe_curve_length
             self.fe_curve_channels = fe_curve_channels


         def forward(self, x_t, time_emb, protein_embedding, condition_embedding):
             # Flatten inputs and concatenate for dummy linear layers
             batch_size = x_t.size(0)
             x_t_flat = x_t.view(batch_size, -1)
             protein_flat = protein_embedding.view(batch_size, -1)
             condition_flat = condition_embedding.view(batch_size, -1)
             time_flat = time_emb.view(batch_size, -1)

             combined_input = torch.cat([x_t_flat, protein_flat, condition_flat, time_flat], dim=-1)

             h = self.fc1(combined_input)
             output_flat = self.fc2(h)

             # Reshape output to match expected shape
             output = output_flat.view(batch_size, self.fe_curve_length, self.fe_curve_channels)
             return output # Simulate predicting noise


    class DummyDiffusionModel(nn.Module):
         def __init__(self, fe_curve_length, fe_curve_channels, protein_input_dim, protein_embed_dim, protein_encoding_type, condition_input_dim, condition_embed_dim, time_embed_dim, model_channels, num_diffusion_steps, beta_schedule):
             super().__init__()
             self.fe_curve_length = fe_curve_length
             self.fe_curve_channels = fe_curve_channels
             self.num_diffusion_steps = num_diffusion_steps

             # Simplified diffusion schedule for dummy test
             self.betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
             self.alphas = 1.0 - self.betas
             self.alphas_bar = torch.cumprod(self.alphas, dim=0)
             self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
             self.sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - self.alphas_bar)

             # Instantiate dummy encoders and denoising model
             self.protein_encoder = DummyProteinEncoder(protein_input_dim, protein_embed_dim, protein_encoding_type)
             self.condition_encoder = DummyConditionEncoder(condition_input_dim, condition_embed_dim)
             self.time_embedding = DummyTimeEmbedding(time_embed_dim)
             self.denoising_model = DummyDenoisingModel(
                 fe_curve_length, fe_curve_channels, protein_embed_dim, condition_embed_dim, time_embed_dim, model_channels
             )

         def sample_timesteps(self, batch_size, device):
            return torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)

         def forward_diffusion(self, x_0, t, noise=None):
             if noise is None: noise = torch.randn_like(x_0)
             sqrt_alphas_bar_t = self.sqrt_alphas_bar[t].view(-1, 1, 1).to(x_0.device) # Ensure on correct device
             sqrt_one_minus_alphas_bar_t = self.sqrt_one_minus_alphas_bar[t].view(-1, 1, 1).to(x_0.device) # Ensure on correct device
             x_t = sqrt_alphas_bar_t * x_0 + sqrt_one_minus_alphas_bar_t * noise
             return x_t, noise

         def forward(self, x_0, sequence_data, conditions):
             batch_size = x_0.size(0)
             device = x_0.device
             t = self.sample_timesteps(batch_size, device)
             x_t, noise = self.forward_diffusion(x_0, t)

             protein_emb = self.protein_encoder(sequence_data)
             condition_emb = self.condition_encoder(conditions)
             time_emb = self.time_embedding(t)

             predicted_noise = self.denoosing_model(x_t, time_emb, protein_emb, condition_emb) # Typo fix: denoising_model
             # Corrected call:
             predicted_noise = self.denoising_model(x_t, time_emb, protein_emb, condition_emb)


             return x_t, t, predicted_noise, noise

         @torch.no_grad()
         def generate(self, sequence_data, conditions, num_samples=1, device='cpu'):
             # Simplified generation for dummy model
             logging.warning("Using dummy generate method in DummyDiffusionModel.")
             batch_size = len(sequence_data) if isinstance(sequence_data, list) else sequence_data.size(0)

             # Simulate returning random data with correct shape
             generated_curves = torch.randn(batch_size, self.fe_curve_length, self.fe_curve_channels, device=device)
             return generated_curves


    # Dummy parameters
    fe_len = 200
    seq_input_dim = 100
    seq_embed_dim = 128
    cond_input_dim = 2
    cond_embed_dim = 64
    time_embed_dim = 128
    model_channels = 256
    num_diff_steps = 100
    num_epochs = 5
    batch_sz = 16

    # Create dummy data and dataloaders
    dummy_train_dataset = DummyDataset(num_samples=1000, fe_len=fe_len, seq_dim=seq_input_dim, cond_dim=cond_input_dim)
    dummy_val_dataset = DummyDataset(num_samples=200, fe_len=fe_len, seq_dim=seq_input_dim, cond_dim=cond_input_dim)
    dummy_train_dataloader = DataLoader(dummy_train_dataset, batch_size=batch_sz, shuffle=True)
    dummy_val_dataloader = DataLoader(dummy_val_dataset, batch_size=batch_sz, shuffle=False)

    # Create dummy model
    dummy_model = DummyDiffusionModel(
        fe_curve_length=fe_len,
        fe_curve_channels=1,
        protein_input_dim=seq_input_dim,
        protein_embed_dim=seq_embed_dim,
        protein_encoding_type='pretrained_embeddings',
        condition_input_dim=cond_input_dim,
        condition_embed_dim=cond_embed_dim,
        time_embed_dim=time_embed_dim,
        model_channels=model_channels,
        num_diffusion_steps=num_diff_steps,
        beta_schedule='linear'
    )

    # Define dummy optimizer and loss configurations
    optimizer_cfg = {'name': 'adamw', 'lr': 1e-4, 'weight_decay': 1e-5}
    scheduler_cfg = {'name': 'step', 'params': {'step_size': 2, 'gamma': 0.5}} # Halve LR every 2 epochs
    # scheduler_cfg = {'name': 'reduceonplateau', 'params': {'mode': 'min', 'factor': 0.5, 'patience': 3}}
    # scheduler_cfg = {'name': 'none'}

    loss_cfg = {
        'noise_weight': 1.0,
        # 'mech_prop_weight': 0.1, # Uncomment to test mech prop loss (placeholder)
        # 'curve_match_weight': 0.1, # Uncomment to test curve match loss (placeholder)
        # 'mech_prop_weights': {'unfolding_energy': 0.5, 'max_force': 0.5} # If mech_prop_weight > 0
    }

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Instantiate and run the Trainer
    try:
        trainer = Trainer(
            model=dummy_model,
            train_dataloader=dummy_train_dataloader,
            val_dataloader=dummy_val_dataloader,
            optimizer_config=optimizer_cfg,
            scheduler_config=scheduler_cfg,
            loss_config=loss_cfg,
            epochs=num_epochs,
            device=device,
            log_dir='dummy_runs/',
            checkpoint_dir='dummy_checkpoints/',
            save_interval=2, # Save every 2 epochs
            eval_interval=1 # Evaluate every epoch
        )
        print("Trainer instantiated successfully.")

        # Clean up previous dummy runs/checkpoints
        # import shutil
        # if os.path.exists('dummy_runs'): shutil.rmtree('dummy_runs')
        # if os.path.exists('dummy_checkpoints'): shutil.rmtree('dummy_checkpoints')


        trainer.train()

        # Example of loading a checkpoint
        # print("\n--- Testing loading checkpoint ---")
        # latest_checkpoint = os.path.join('dummy_checkpoints', f'checkpoint_epoch_{num_epochs}.pt')
        # if os.path.exists(latest_checkpoint):
        #      print(f"Attempting to load from {latest_checkpoint}")
        #      # Need a new Trainer instance or reset state if loading into the same one
        #      # For simplicity, let's just demonstrate the load_checkpoint call
        #      try:
        #          dummy_model_loaded = DummyDiffusionModel(
        #             fe_curve_length=fe_len, fe_curve_channels=1, protein_input_dim=seq_input_dim, protein_embed_dim=seq_embed_dim, protein_encoding_type='pretrained_embeddings',
        #             condition_input_dim=cond_input_dim, condition_embed_dim=cond_embed_dim, time_embed_dim=time_embed_dim, model_channels=model_channels, num_diffusion_steps=num_diff_steps, beta_schedule='linear'
        #          ).to(device)
        #          dummy_optimizer_loaded = get_optimizer(dummy_model_loaded, **optimizer_cfg)
        #          dummy_trainer_loaded = Trainer(
        #              model=dummy_model_loaded, train_dataloader=dummy_train_dataloader, val_dataloader=dummy_val_dataloader,
        #              optimizer_config=optimizer_cfg, scheduler_config=scheduler_cfg, loss_config=loss_cfg, epochs=num_epochs,
        #              device=device, log_dir='dummy_runs_load/', checkpoint_dir='dummy_checkpoints_load/'
        #          )
        #          start_epoch_loaded = dummy_trainer_loaded.load_checkpoint(latest_checkpoint)
        #          print(f"Successfully loaded checkpoint. Next epoch would be {start_epoch_loaded}.")
        #      except Exception as e:
        #          print(f"Error during checkpoint loading test: {e}")
        # else:
        #      print(f"Checkpoint {latest_checkpoint} not found to test loading.")


    except Exception as e:
        print(f"An error occurred during Trainer test: {e}")
        import traceback
        traceback.print_exc()