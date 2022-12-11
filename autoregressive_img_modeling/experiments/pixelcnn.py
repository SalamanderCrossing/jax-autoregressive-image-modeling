import mate
from ..data_loaders.mnist import get_loaders
from ..models.pixelcnn import PixelCNN
from ..trainers.trainer import TrainerModule


train_loader, val_loader, test_loader = get_loaders(
    train_batch_size=128,
    test_batch_size=128,
)

model = PixelCNN(c_in=1, c_hidden=64)
trainer = TrainerModule(
    model, next(iter(train_loader))[0], mate.default_checkpoint_location, mate.save_dir
)
if mate.is_train:
    _, result = trainer.train_model(train_loader, val_loader, num_epochs=10)
    mate.result(result)
elif mate.is_test:
    result = trainer.test_model(test_loader)
    mate.result(result)
