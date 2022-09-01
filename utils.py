from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json
import torch


class Meter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0
        self.obs = 0
        self.mean = 0.0

    def update(self, loss: float, obs: int):
        self.value += loss * obs
        self.obs += obs
        self.mean = self.value / self.obs

    @property
    def avg_loss(self):
        return self.mean


class Logger(object):
    def __init__(self):
        self.logging = []
        self.writer = SummaryWriter()
        self.epoch = 1

    def update(self, info: dict):
        self.logging.append(info)
        for name, value in info.items():
            self.writer.add_scalar(name, value, self.epoch)
        self.epoch += 1

    def save(self, path: str):
        with open(path, "w") as fileLink:
            fileLink.write(json.dumps(self.logging))


def get_lr(optimizer) -> float:
    for i in optimizer.param_groups:
        return i["lr"]


def validate(model, loaders: dict, device: str = "cuda"):
    result = {}
    meter = Meter()
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for name, loader in loaders.items():
        top1 = 0
        top5 = 0
        obs = 0
        for batch in tqdm(loader):
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            obs += features.size(0)
            with torch.no_grad():
                logits = model(features)
                loss = criterion(logits, targets)
                meter.update(loss.item(), features.size(0))
                preds = torch.nn.functional.softmax(logits, dim=1)
                top1_correct, top5_correct = get_top1_top5(preds, targets)
                top1 += top1_correct
                top5 += top5_correct
        result[name] = {
            "top1": round(top1 / obs, 2),
            "top5": round(top5 / obs, 2),
            "loss": meter.avg_loss,
        }
        meter.reset()

    return result


def get_top1_top5(preds, targets):
    top1_indices = torch.topk(preds, 1, dim=1)[1]
    top5_indices = torch.topk(preds, 5, dim=1)[1]
    top1_correct, top5_correct = 0, 0
    for target, top1_index, top5_index in zip(
        targets.tolist(), top1_indices.tolist(), top5_indices.tolist()
    ):
        top1_correct += target in top1_index
        top5_correct += target in top5_index

    return top1_correct, top5_correct
