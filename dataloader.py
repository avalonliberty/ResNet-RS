from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from glob import glob


class ImageNet(Dataset):
    # https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data
    # Downloading LOC_synset_mapping.txt file from the URL above
    # for getting synset and label name mapping

    def __init__(self, root_path: str, label_path: str, mode: str = "train"):
        assert mode in ["train", "val"], "mode must be either train or val"
        folder_path = f"{root_path}/{mode}"
        self.image_list = glob(f"{folder_path}/n*/*")
        self.transformer = transforms.Compose(
            [
                transforms.RandAugment(num_ops=2, magnitude=15),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self._get_labels(label_path)

    def _get_labels(self, label_path: str) -> dict:
        synset_mapper = {}
        label_mapper = {}
        with open(label_path, "r") as fileLink:
            rows = fileLink.readlines()
            for target_id, row in enumerate(rows):
                row = row.replace("\n", "")
                collection = row.split(" ")
                synset_id = collection[0]
                name = " ".join(collection[1:])
                synset_mapper[synset_id] = name
                label_mapper[name] = target_id

        self.synset_mapper = synset_mapper
        self.label_mapper = label_mapper

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, index: int) -> dict:
        image_path = self.image_list[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transformer(image)
        synset_id = Path(image_path).parent.name
        label = self.synset_mapper[synset_id]
        target_id = self.label_mapper[label]

        return {"features": image, "targets": target_id, "labels": label}
