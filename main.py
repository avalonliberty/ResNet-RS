from argparse import ArgumentParser
from torch.optim import SGD, lr_scheduler
from torch.cuda.amp import GradScaler
from torch.nn import CrossEntropyLoss
from resnetrs import ResNet
from dataloader import ImageNet
from pathlib import Path
from torch.utils.data import DataLoader
from utils import Meter, Logger, get_lr, validate
from tqdm import tqdm
import torch

argparser = ArgumentParser("resnetrs training")
argparser.add_argument("--model", default="resnetrs50")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 350
BATCH_SIZE = 64
LABEL_SMOOTH = 0.1


def get_optimizer(params: list, batch_size: int):
    lr = 0.1 / batch_size  # following ResNet-RS
    optimizer = SGD(params, lr, momentum=0.9, weight_decay=4e-5)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5, 2)

    return optimizer, scheduler


def train(model, dataloaders, validate_step: int = None):
    trainloader = dataloaders["train"]
    if validate_step:
        valloader = dataloaders["val"]
    num_batches = len(trainloader)
    optimizer, scheduler = get_optimizer(model.parameters(), BATCH_SIZE)
    scaler = GradScaler()
    criterion = CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    meter = Meter()
    logger = Logger()
    for epoch in range(1, EPOCHS + 1):
        current_lr = get_lr(optimizer)
        for iters, batch in enumerate(tqdm(trainloader)):
            features = batch["features"].to(DEVICE)
            targets = batch["targets"].to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE):
                predictions = model(features)
                loss = criterion(predictions, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step(epoch + iters / num_batches)
            meter.update(loss.item(), features.size(0))
        record = {"lr": current_lr, "training_loss": loss.item()}
        if validate_step:
            if epoch % validate_step == 0:
                validation = validate(
                    model, {"train": trainloader, "val": valloader}, DEVICE
                )
                for train_type in ["train", "val"]:
                    valid_result = validation[train_type]
                    for metric_name, metric in valid_result.items():
                        record[f"{train_type}_{metric_name}"] = metric
        logger.update(record)
        meter.reset()
    torch.save(model, "model.pth")


if __name__ == "__main__":
    args = argparser.parse_args()
    model_name = args.model
    model = ResNet.build_model(model_name).to(DEVICE)
    trainset = ImageNet(
        f"{Path.cwd().parent}/imageNet/2012",
        f"{Path.cwd().parent}/imageNet/LOC_synset_mapping.txt",
        mode="train",
    )
    valset = ImageNet(
        f"{Path.cwd().parent}/imageNet/2012",
        f"{Path.cwd().parent}/imageNet/LOC_synset_mapping.txt",
        mode="val",
    )
    trainloader = DataLoader(trainset, BATCH_SIZE, shuffle=True, num_workers=6)
    valloader = DataLoader(valset, BATCH_SIZE, shuffle=False, num_workers=6)
    train(model, {"train": trainloader, "val": valloader}, validate_step=10)
