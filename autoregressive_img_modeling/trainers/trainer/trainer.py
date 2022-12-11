from flax import linen as nn
from flax.training import train_state, checkpoints
import optax
from torch.utils.data import DataLoader
from tqdm import tqdm
import jax.numpy as jnp
from jax import random
import jax
import os
from torch.utils.tensorboard import SummaryWriter


class TrainerModule:
    def __init__(
        self,
        model: nn.Module,
        exmp_imgs: jnp.ndarray,
        checkpoint_path: str,
        log_dir: str,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        """
        Module for summarizing all training functionalities for the PixelCNN.
        """
        super().__init__()
        self.lr = lr
        self.seed = seed
        # Create empty model. Note: no parameters yet
        self.model = model  # PixelCNN(c_in=c_in, c_hidden=c_hidden)
        # Prepare logging
        # self.log_dir = checkpoint_path  # os.path.join(CHECKPOINT_PATH, self.model_name)
        self.checkpoint_path = checkpoint_path
        self.logger = SummaryWriter(log_dir=log_dir)  # self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)

    def create_functions(self):
        # Training function
        def train_step(state, batch):
            imgs, _ = batch
            loss_fn = lambda params: state.apply_fn(params, imgs)
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss

        # Eval function
        def eval_step(state, batch):
            imgs, _ = batch
            loss = state.apply_fn(state.params, imgs)
            return loss

        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_imgs):
        # Initialize model
        init_rng = random.PRNGKey(self.seed)
        params = self.model.init(init_rng, exmp_imgs)
        self.state = train_state.TrainState(
            step=0, apply_fn=self.model.apply, params=params, tx=None, opt_state=None
        )

    def init_optimizer(self, num_epochs: int, num_steps_per_epoch: int):
        # Initialize learning rate schedule and optimizer
        lr_schedule = optax.exponential_decay(
            init_value=self.lr, transition_steps=num_steps_per_epoch, decay_rate=0.99
        )
        optimizer = optax.adam(lr_schedule)
        # Initialize training state
        self.state = train_state.TrainState.create(
            apply_fn=self.state.apply_fn, params=self.state.params, tx=optimizer
        )

    def train_model(
        self, train_loader: DataLoader, val_loader: DataLoader, num_epochs=200
    ):
        if not self.checkpoint_exists():
            # Train model for defined number of epochs
            # We first need to create optimizer and the scheduler for the given number of epochs
            self.init_optimizer(num_epochs, len(train_loader))
            # Track best eval bpd score.
            best_eval = 1e6
            for epoch_idx in tqdm(range(1, num_epochs + 1)):
                self.train_epoch(train_loader, epoch=epoch_idx)
                if epoch_idx % 1 == 0:
                    eval_bpd = self.eval_model(val_loader)
                    self.logger.add_scalar("val/bpd", eval_bpd, global_step=epoch_idx)
                    if eval_bpd <= best_eval:
                        best_eval = eval_bpd
                        self.save_model(step=epoch_idx)
                self.logger.flush()

        self.load_model()
        val_bpd = self.eval_model(val_loader)
        # Bind parameters to model for easier inference
        self.model_bd = self.model.bind(self.state.params)
        return self, {"val_bpd": val_bpd}  # , "test_bpd": val_bpd}

    def test_model(self, data_loader: DataLoader):
        assert (
            self.checkpoint_exists()
        ), "No checkpoint found. Please train model first."
        self.load_model()
        test_bpd = self.eval_model(data_loader)
        return test_bpd

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        # Train model for one epoch, and log avg bpd
        avg_loss = 0
        for batch in tqdm(train_loader, desc="Training", leave=False):
            self.state, loss = self.train_step(self.state, batch)
            avg_loss += loss
        avg_loss /= len(train_loader)
        self.logger.add_scalar("train/bpd", avg_loss.item(), global_step=epoch)

    def eval_model(self, data_loader: DataLoader):
        # Test model on all images of a data loader and return avg bpd
        avg_bpd, count = 0, 0
        for batch in data_loader:
            bpd = self.eval_step(self.state, batch)
            avg_bpd += bpd * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_bpd = (avg_bpd / count).item()
        return eval_bpd

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(
            ckpt_dir=self.checkpoint_path,
            target=self.state.params,
            step=step,
            overwrite=True,
        )

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=self.checkpoint_path, target=None
            )
        else:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=self.checkpoint_path,
                target=None,
            )
        self.state = train_state.TrainState.create(
            apply_fn=self.state.apply_fn,
            params=state_dict,
            tx=self.state.tx if self.state.tx else optax.sgd(0.1),  # Default optimizer
        )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(self.checkpoint_path)
