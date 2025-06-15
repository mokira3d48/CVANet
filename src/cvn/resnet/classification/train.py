#!/usr/bin/env python3
#-*- encoding: utf8 -*-

"""
+=============================================================================+
|          ALEX-NET CLASSIFICATION TRAINING IMPLEMENTATION                    |
+=============================================================================+


MIT License

Copyright (c) 2025 Doctor Mokira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = '0.1.0'
__auther__ = 'Doctor Mokira'

import os
import json
import argparse
import logging
import time
from shutil import copy

import yaml
import numpy as np
import matplotlib.pyplot as plt
from jinja2.optimizer import optimize
from tqdm import tqdm
from PIL import Image, ExifTags
# from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics as m

import torch
# import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data
from torch.utils.data import Dataset as BaseDataset
import torchvision.transforms.functional as TF
# from torchvision import transforms
from torchinfo import summary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    format='%(asctime)s - - \033[95m%(levelname)s\033[0m - %(message)s',
    handlers=[
        logging.FileHandler("resnet_class_train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def set_seed(seed, device):
    """
    Set seeds for reproducibility

    :param seed: An integer value to define the seed for random generator.
    :param device: The selected device.

    :type seed: `int`
    :type device: torch.device
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if device.type == 'cuda':
        # Also set the deterministic flag for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


###############################################################################
#  ALEX-NET MODEL IMPLEMENTATION
###############################################################################

class ModelConfig:

    def __init__(
        self, img_size=(224, 224), img_channels=3, num_classes=10,
        dropout_prob=0.2, in_channels=64,
        mean=[0.485, 0.456, 0.406],  # noqa
        std=[0.229, 0.224, 0.225],  # noqa
    ):
        self.img_size = img_size
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.in_channels = in_channels
        self.mean = mean
        self.std = std

    def state_dict(self):
        """
        Returns the dictionary of variables associated with its values
        :rtype: typing.Dict[str, object]
        """
        return self.__dict__

    def save(self, file_path):
        """
        Save the attribute values into a YAML file

        :param file_path: The path to the YAML file
        :type file_path: str
        """
        config_data = self.state_dict()
        with open(file_path, mode='w', encoding='utf-8') as f:
            yaml.dump(config_data, f)

    def load_state_dict(self, state_dict):
        """
        Method to load state dict and update model config value

        :param state_dict: The dictionary of model config attribute
        :type state_dict: typing.Dict[str, object]
        """
        self.__dict__.update(state_dict)

    def load(self, file_path):
        """
        Load the attribute values from a YAML file

        :param file_path: The YAML file where the model config data is stored
        :type file_path: str
        """
        with open(file_path, mode='r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            self.load_state_dict(config_data)


class Block(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            identity_downsample=None,
            stride=1
    ):
        super().__init__()

        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride,
            padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, (out_channels * self.expansion), kernel_size=1,
            stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.identity_downsample:
            identity = self.identity_downsample(identity)

        x = x + identity
        return self.relu(x)


class ResNet50(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        self.img_channels = self.config.img_channels
        self.num_classes = self.config.num_classes
        self.in_channels = self.config.in_channels

        # self.conv1 = nn.Conv2d(
        #     self.img_channels, self.in_channels, kernel_size=7, stride=2,
        #     padding=3
        # )
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU()
        #
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        # # ResNet Layers
        # self.layer1 = self._make_layer(3, 64, stride=1)
        # self.layer2 = self._make_layer(4, 128, stride=2)
        # self.layer3 = self._make_layer(6, 256, stride=2)
        # self.layer4 = self._make_layer(3, 512, stride=2)
        # #: 2048 channels at the end

        self.backbone = nn.ModuleList([
            nn.Conv2d(
                self.img_channels, self.in_channels, kernel_size=7, stride=2,
                padding=3
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # ResNet Layers
            self._make_layer(3, 64, stride=1),
            self._make_layer(4, 128, stride=2),
            self._make_layer(6, 256, stride=2),
            self._make_layer(3, 512, stride=2),
            #: 2048 channels at the end

            nn.AdaptiveAvgPool2d((1, 1))
        ])

        self.fc = nn.Linear(512 * 4, self.num_classes)

    def _make_layer(self, num_residual, out_channels, stride):
        identity_downsample = None
        layers = nn.ModuleList()

        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, out_channels * 4, kernel_size=1,
                    stride=stride),
                nn.BatchNorm2d(out_channels * 4))

        layers.append(Block(
            self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_residual - 1):
            layers.append(Block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass method

        :param x: input image with dim: [batch_size, c, H, W].
        :returns: The model prediction formated as a tensor with dim:
          [batch_size, num_classes].

        :type x: torch.Tensor
        :rtype: torch.Tensor
        """
        for module in self.backbone:
            x = module(x)

        x = x.reshape(x.shape[0], -1)
        out = self.fc(x)
        return out


class Input(nn.Module):

    def __init__(
        self, img_channels=3, img_size=(224, 224),
        mean=[0.485, 0.456, 0.406],  # noqa
        std=[0.229, 0.224, 0.225],  # noqa
        device=None
    ):
        super().__init__()
        assert img_channels in (1, 3), (
            "Either image channels is equal to 3 or equal to 1"
            f" Never equal to {img_channels}"
        )
        self.img_channels = img_channels
        self.img_size = list(img_size)
        self.mean = mean
        self.std = std
        self.device = device if device else torch.device('cpu')

    def forward(self, x):
        """
        Preprocessing method
        --------------------

        :param x: [batch_size, w, h, img_channels];
        :returns: [batch_size, img_channels, h, w]

        :type x: torch.Tensor
        :rtype: torch.Tensor
        """
        assert x.shape[-1] in (1, 3), (
            f"Expected 1 or 3 as image channels, but {x.shape[-1]} is got."
        )

        # [batch_size, img_channels, h, w]
        x = x.permute((0, 3, 2, 1))
        x = x.contiguous()

        if x.shape[-2:] != self.img_size:
            x = TF.resize(x, self.img_size)

        # RGB, Gray scale conversion
        if x.shape[1] == 1 and self.img_channels == 3:
            x = torch.cat([x, x, x], dim=1)
            x = TF.normalize(x, self.mean, self.std)
        elif x.shape[1] == 3 and self.img_channels == 1:
            x = TF.rgb_to_grayscale(x, num_output_channels=1)

        # Normalization
        x = x / 255.0
        x = x.to(torch.float32)
        return x


class Output(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Post-processing method
        ----------------------

        :param x: [batch_size, num_classes];
        :returns: tuple of two tensors of size [batch_size,],
          the first tensor contains the class ids predicted
          and the second tensor contains the softmax confidences.

        :type x: torch.Tensor
        :rtype: typing.Tuple[torch.Tensor, torch.Tensor]
        """
        probs = torch.softmax(x, dim=-1)  # [n, num_classes]
        class_ids = torch.argmax(probs, dim=-1)  # [n,]
        confidences = torch.max(probs, dim=-1).values  # [n,]
        return class_ids, confidences


class Model(ResNet50):

    def __init__(self, config):
        super().__init__(config)
        self.input = Input(
            img_channels=self.config.img_channels,
            img_size=self.config.img_size,
            mean=self.config.mean,
            std=self.config.std,
        )
        self.output = Output()

    def device(self):
        return next(self.parameters()).device

    def summary(self, batch_size=1):
        """
        Method of model summary

        :param batch_size: The batch size

        :type batch_size: int
        """
        model_device = self.device()
        B = batch_size
        img_channels = self.config.img_channels
        img_size = self.config.img_size
        input_image = torch.randn((B, img_channels, *img_size))
        input_image = input_image.to(model_device)
        results = summary(self, input_data=input_image)
        return results

    def weights(self):
        """Return the model weights"""
        state_dict = self.state_dict()
        return state_dict

    def save(self, file_path):
        """
        Function to save model weights into file.

        :params file_path: The model file path.
        :type file_path: `str`
        """
        os.makedirs(file_path, exist_ok=True)
        model_file = os.path.join(file_path, 'weights.pth')
        param_file = os.path.join(file_path, 'config.yaml')

        model_weights = self.weights()
        torch.save(model_weights, model_file)
        self.config.save(param_file)

    @classmethod
    def load(cls, file_path):
        """
        Static method to load model weights from file

        :param file_path: The path to folder that contents the model weight
          file and the model config file as YAML format
        :returns: An instance of the model with its weights correctly loaded.

        :type file_path: str
        :rtype: Model
        """
        model_file = os.path.join(file_path, 'weights.pth')
        param_file = os.path.join(file_path, 'config.yaml')
        if not os.path.isfile(param_file):
            raise FileNotFoundError(
                f"No such model config file at {param_file}.")
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"No such model file at {model_file}.")

        hparams = ModelConfig()
        hparams.load(param_file)
        instance = cls(hparams)
        weights = torch.load(model_file, weights_only=True, map_location='cpu')
        instance.load_state_dict(weights)
        logger.info("Model weights loaded successfully!")
        return instance


def fine_tune_model(
    model, num_new_classes, lr, freeze_feature_layers=True,
    weight_decay=0.005
):
    """
    Load a pre-trained model and modify it for fine-tuning
    on a new dataset with different number of classes.

    :param model: Instance of the pre-trained model.
    :param num_new_classes: Number of classes in the new dataset.
    :param lr: The base learning rate will be used to config optimizer
        with the deference layers learning rate.
    :param freeze_feature_layers: Whether to freeze the feature
        extraction layers.
    :param weight_decay: The weight decay value.

    :type model: Model
    :type num_new_classes: `int`
    :type lr: `float`
    :type freeze_feature_layers: `bool`
    :type weight_decay: float

    :returns:
        model: Modified model ready for fine-tuning
        optimizer: Configured optimizer for fine-tuning
    :rtype: typing.Tuple[Model, torch.optim.Optimizer]
    """
    # Load the checkpoint
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Model.load(pretrained_model_path)

    model.fc = nn.Linear(512 * 4, num_new_classes)
    model.config.num_classes = num_new_classes

    if freeze_feature_layers:
        optimizer = optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        backbone_params = list(model.backbone.parameters())
        # post_backbone_params = list(model.post_backbone.parameters())
        fc_params = list(model.fc.parameters())
        optimizer = optim.Adam(
            [
                {'params': backbone_params, 'lr': lr * 0.01},
                #: Very low learning rate for frozen feature layers

                # {'params': post_backbone_params, 'lr': lr * 0.1},
                #: Medium learning rate for FC1 and FC2

                {'params': fc_params, 'lr': lr}
                #: High learning rate for the new classification layer
            ],
            weight_decay=weight_decay
        )
    return model, optimizer


###############################################################################
# DATASET IMPLEMENTATION
###############################################################################

def fix_image_rotation(image_path, output_path=None):
    """
    Fix image rotation based on EXIF orientation data

    :param image_path: Path to the input image
    :param output_path: Path to save the corrected image (optional)
    :returns: Corrected image object

    :type image_path: `str`
    :type output_path: `str`
    :rtype: numpy.ndarray
    """
    try:
        # Open the image
        image = Image.open(image_path)

        # Get EXIF data
        exif = image._getexif()

        if exif is not None:
            # Find orientation tag
            orientation_key = None
            for tag, value in ExifTags.TAGS.items():
                if value == 'Orientation':
                    orientation_key = tag
                    break

            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]

                # Apply rotation based on orientation value
                if orientation == 2:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 4:
                    image = image.transpose(Image.FLIP_TOP_BOTTOM)
                elif orientation == 5:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    image = image.rotate(90, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 7:
                    image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)

        # Save the corrected image if output path is provided
        if output_path:
            # Remove EXIF data to prevent further rotation issues
            image.save(output_path, quality=95, optimize=True)

        return image

    except Exception as e:
        print(f"Error processing image: {e}")
        return None


class Dataset(BaseDataset):
    """
    Dataset implementation

    :arg inputs: The list of features representing the image files
    :arg targets: The list of the targets
    :arg class_names: The list of available class names
    :arg transform: The pipeline of image transformation

    :type inputs: typing.List[str]
    :type targets: typing.List[str]
    :type targets:
    """
    def __init__(
        self, inputs, targets, class_names, img_size=(224, 224), transform=None
    ):
        self.samples = [(x, y) for x, y in zip(inputs, targets)]
        self.class_names = class_names
        self.img_size = img_size
        self.transform = transform

    @classmethod
    def build(cls, data_dir, img_size=(224, 224), transform=None):
        """
        Class method to build a dataset from data samples
        contained on data dir

        :param data_dir: The path to the data directory
        :param img_size: The image size will be used to resize all images
        :param transform: The pipeline of data transformation
        :returns: An instance of the Dataset.

        :type data_dir: `str`
        :type img_size: typing.Tuple[int, int]
        :type transform: torchvision.transforms.Compose
        :rtype: Dataset
        """
        paths = []
        classes = []
        class_names = []
        folder_names = os.listdir(data_dir)
        folder_names = sorted(folder_names)

        loader = tqdm(folder_names, total=len(folder_names))
        for index, folder_name in enumerate(loader):
            class_name = folder_name
            class_idx = index

            class_names.append(class_name)
            folder_path = os.path.join(data_dir, folder_name)
            file_names = os.listdir(folder_path)
            file_count = len(file_names)
            loader.set_description(
                f"Loading of {file_count} from {folder_name}"
            )
            for fid, file_name in enumerate(file_names):
                is_image_file = (
                    file_name.endswith(".png") or file_name.endswith(".jpg")
                )
                if not is_image_file:
                    continue
                file_path = os.path.join(folder_path, file_name)
                #fix_image_rotation(file_path, file_path)

                paths.append(file_path)
                classes.append(class_idx)

                loader.set_postfix(files=f"{(fid + 1)}/{file_count}")

            loader.write(f"Class {class_name} of {file_count} is processed.")

        loader.set_description("Done")

        instance = cls(
            paths, classes, class_names, img_size=img_size,
            transform=transform
        )
        return instance

    def __len__(self):
        # return 1000
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]

        img = Image.open(path).convert('RGB')
        img = img.resize(self.img_size)

        if self.transform:
            img = self.transform(img)

        img = np.array(img)
        img = torch.as_tensor(img)
        target = torch.as_tensor(target)

        return img, target


###############################################################################
# METRICS IMPLEMENTATION
###############################################################################


class AvgMeter:

    def __init__(self, val=0):
        self.total = val
        self.count = 0

    def reset(self):
        self.total = 0.0
        self.count = 0

    def __add__(self, other):
        """
        Add function

        :type other: `float`|`int`
        """
        self.total += other
        self.count += 1
        return self

    def __radd__(self, other):
        return self.__add__(other)

    def avg(self):
        if self.count > 0:
            return self.total / self.count
        else:
            return 0.0

    def __str__(self):
        return str(self.total)


class Metric:

    @classmethod
    def load(cls, state_dict, new_num_epochs=None):
        """
        Method to load metric data from state dict

        :param state_dict: The state dict which contents the metrics data
        :param new_num_epochs: The new value of number of epochs
        :return: An instance of Metric with loaded data. None value
          is returned, when the state dict provided is empty or None.

        :type state_dict: `dict`
        :type new_num_epochs: `int`
        :rtype: Metric
        """
        if not state_dict:
            return
        num_epochs = state_dict['num_epochs']
        num_channels = state_dict['num_channels']
        if new_num_epochs and new_num_epochs > num_epochs:
            instance = cls(new_num_epochs, num_channels)
        else:
            instance = cls(num_epochs, num_channels)
        instance.channels = state_dict['channels']

        del state_dict['num_epochs']
        del state_dict['num_channels']
        del state_dict['channels']

        for name, values in state_dict.items():
            metric = instance.__dict__[name]
            for i in range(instance.num_channels):
                metric[i, :num_epochs] = values[i, :num_epochs]
            # logger.info(f"\n\t{name}: {metric}")
        logger.info("Metric state dict is loaded successfully")
        return instance

    def __init__(self, num_epochs, num_channels=2):
        self.num_epochs = num_epochs
        self.num_channels = num_channels

        self.channels = [f"ch_{c}" for c in range(self.num_channels)]

        self.cross_entropy_loss = self.new_buffer()
        self.precision_score = self.new_buffer()
        self.recall_score = self.new_buffer()
        self.f1_score = self.new_buffer()

    def new_buffer(self):
        bf = np.zeros((self.num_channels, self.num_epochs))
        bf[:] = np.nan
        return bf

    def channel_id(self, name):
        if name not in self.channels:
            raise ValueError(f"No channel name '{name}' found")
        return self.channels.index(name)

    def __setitem__(self, epoch, metric_values):
        if not (0 <= epoch < self.num_epochs):
            logger.warning(
                "Epoch indexed is out of range. The max epoch indexable"
                f" is {self.num_epochs}"
            )
            return

        for m_name, m_values in metric_values.items():
            if not hasattr(self, m_name):
                logger.warning(f"Metric named {m_name} is not defined")
                continue
            if isinstance(m_values, dict):
                for chn, m_value in m_values.items():
                    chi = self.channel_id(chn)
                    self.__dict__[m_name][chi, epoch] = m_value
            elif isinstance(m_values, list):
                if len(m_values) != self.num_channels:
                    raise ValueError(
                        "The length of metric values list must be equal"
                        f"to number channels ({self.num_channels})"
                    )
                for i in range(self.num_channels):
                    self.__dict__[m_name][i, epoch] = m_values[i]

    def state_dict(self):
        """
        Method that is used to return the state dict

        :rtype: `dict`
        """
        return self.__dict__

    def plot(self, save_path):
        """
        Plot and save training curves
        """
        plt.figure(figsize=(12, 10))

        # Plot losses
        plt.subplot(2, 1, 1)
        plt.plot(self.epochs, self.cross_entropy_losses, label='Train Loss')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        # Plot angle errors
        plt.subplot(2, 1, 2)
        plt.plot(epochs, self.train_angles, label='Train Angle Error (rad)')
        plt.plot(epochs, self.val_angles, label='Validation Angle Error (rad)')
        plt.xlabel('Epoch')
        plt.ylabel('Angle Error (radians)')
        plt.title('Training and Validation Angle Error')
        plt.legend()
        plt.grid(True)

        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


###############################################################################
# TRAINING PROCESS
###############################################################################


class Training(Model):

    @classmethod
    def load(cls, file_path=None, checkpoint_file=None):
        """
        Static method to load model state dict from files or checkpoints

        :param file_path: The model file contained the weights and config;
        :param checkpoint_file: The file path to the training checkpoint;
        :returns: The instance of the model.

        :type file_path: `str`
        :type checkpoint_file: `str`
        :rtype: Trainer
        """
        instance = None
        if file_path:
            instance = super().load(file_path)

        if not instance and os.path.isfile(checkpoint_file):
            checkpoint = torch.load(
                checkpoint_file,  weights_only=False, map_location='cpu'
            )

            config = ModelConfig()
            if 'model_config' in checkpoint:
                config.load_state_dict(checkpoint['model_config'])
                logger.info("Model config is loaded from checkpoint!")

            instance = cls(config)
            if 'model_state_dict' in checkpoint:
                instance.load_state_dict(checkpoint['model_state_dict'])
                logger.info("Model state dict is loaded from checkpoint!")

        return instance

    def __init__(self, config=None):
        super().__init__(config)

        self.train_loader = None
        self.val_loader = None

        self.metric = None
        self.cross_entropy = None
        self.optimizer = None
        self.lr_scheduler = None

        self.train_losses = {}
        self.val_losses = {}
        self.cross_entropy_loss = AvgMeter()
        self.precision_score = AvgMeter()
        self.recall_score = AvgMeter()
        self.f1_score = AvgMeter()

        self.gas = 128  # Gradiant Accumulation Steps
        self.seed = 42  # Seed number for random generation
        self.num_epochs = 1
        self.batch_size = 1

        self.gac = 0  # Gradient Accumulation Count

        self.checkpoint_dir = "checkpoints"
        self.resume_ckpt = None
        self.best_model = None
        self.epoch = 0

    def compile(self, args):
        """
        Initialization of training process
        ----------------------------------

        :type args: `argparse.Namespace`
        :rtype: `None`
        """
        model_device = self.device()
        set_seed(args.seed, model_device)
        logger.info(f"Device selected: \033[92m{model_device}\033[0m")

        self.num_epochs = args.epochs
        self.batch_size = args.batch_size
        self.gas = args.gas

        train_ds_dir = args.train_data_dir
        val_ds_dir = args.val_data_dir
        img_size = self.config.img_size

        if not os.path.isdir(train_ds_dir):
            raise FileNotFoundError(
                f"No such training set directory at {train_ds_dir}"
            )
        if not os.path.isdir(val_ds_dir):
            raise FileNotFoundError(
                f"No such validation set directory at {val_ds_dir}"
            )

        train_dataset = Dataset.build(train_ds_dir, img_size=img_size)
        val_dataset = Dataset.build(val_ds_dir, img_size=img_size)

        # Create data loaders
        self.train_loader = data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        self.val_loader = data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False
        )

        # Set loss function
        self.cross_entropy = nn.CrossEntropyLoss()

        # Set up the optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=args.learning_rate,
                weight_decay=args.weight_decay
            )
            logger.info("New instance of optimizer is created.")

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=3, gamma=0.1
        )

        self.resume_ckpt = args.resume
        self.checkpoint_dir = args.checkpoint_dir
        self.best_model = args.best_model

    @staticmethod
    def _update_losses(input_losses, output_losses):
        """
        Update losses value
        """
        for key, val in input_losses.items():
            if key in output_losses:
                output_losses[key] += val
            else:
                new_loss = AvgMeter(val)
                output_losses[key] = new_loss

    @staticmethod
    def print_results(res):
        """
        Function of average meter losses
        """
        string = ''
        for key, val in res.items():
            string += f"{key}: {val:.8f} "
        return string

    @staticmethod
    def add_to_epoch_results(epc_res, new_res):
        for key, val in epc_res.items():
            if key in epc_res:
                epc_res[key].append(new_res[key])
            else:
                epc_res[key] = [new_res[key]]

    def checkpoint(self, **kwargs):
        checkpoint = {
            "model_config": self.config.state_dict(),
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
            "epoch": self.epoch,
            # "train_losses": self.train_losses,
            # "val_losses": self.val_losses,
            "metric_state_dict": self.metric.state_dict(),
            # "best_performance": self.best_performance,
            **kwargs
        }

        file_path1 = os.path.join(self.checkpoint_dir, "checkpoint.pth")
        file_path2 = os.path.join(
            self.checkpoint_dir, f"checkpoint_{self.epoch}.pth"
        )
        torch.save(checkpoint, file_path1)
        copy(file_path1, file_path2)
        logger.info("Checkpoint done successfully!")

        if self.epoch >= 2:
            old_checkpoint_file = os.path.join(
                self.checkpoint_dir, f"checkpoint_{self.epoch - 2}.pth"
            )
            if os.path.isfile(old_checkpoint_file):
                os.remove(old_checkpoint_file)
                logger.info(f"{old_checkpoint_file} checkpoint is removed.")

    def load_checkpoint(self):
        if not self.resume_ckpt:
            return
        if not os.path.isfile(self.resume_ckpt):
            logger.info(f"No such checkpoint file at {self.resume_ckpt}")
            return

        ckpt_data = torch.load(
            self.resume_ckpt, weights_only=False, map_location='cpu'
        )
        self.optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(ckpt_data['lr_scheduler_state_dict'])
        self.epoch = ckpt_data['epoch'] + 1
        # self.train_losses = ckpt_data['train_losses']
        # self.val_losses = ckpt_data['val_losses']
        self.metric = Metric.load(
            ckpt_data.get('metric_state_dict'), self.num_epochs
        )
        # self.best_performance = ckpt_data['best_performance']
        logger.info(f"Checkpoint loaded successfully from {self.resume_ckpt}!")

    def train_step(self, images, targets, write_fn, optimize=False):
        """
        Training method on one batch
        """
        # Forward pass
        images = self.input(images)
        logits = self.forward(images)
        predictions, _ = self.output(logits)

        # Comput losses
        loss = self.cross_entropy(logits, targets)

        # Backward pass
        loss.backward()

        self.cross_entropy_loss += loss.item()

        y_true = targets.cpu().detach().numpy()
        y_pred = predictions.cpu().detach().numpy()

        # Compute metrics
        args = dict(
            y_true=y_true, y_pred=y_pred, average='macro', zero_division=0
        )
        precision = m.precision_score(**args)
        recall = m.recall_score(**args)
        f1 = m.f1_score(**args)

        self.cross_entropy_loss += loss.item()
        self.precision_score += precision
        self.recall_score += recall
        self.f1_score += f1

        self.gac += images.shape[0]
        if self.gac >= self.gas or optimize:
            self.optimizer.step()
            self.optimizer.zero_grad()

            write_fn(
                "\t* Optim step"
                f" - cross_entropy loss: {self.cross_entropy_loss.avg():.8f}"
                f" - precision score: {self.precision_score.avg():5.3f}"
                f" - recall score: {self.recall_score.avg():5.3f}"
                f" - f1 score: {self.f1_score.avg():5.3f}")
            self.gac = 0

    def train_one_epoch(self):
        """
        Method of training on one epoch
        """
        model_device = self.device()
        loss_data = {}

        self.cross_entropy_loss.reset()
        self.precision_score.reset()
        self.recall_score.reset()
        self.f1_score.reset()

        length = len(self.train_loader)
        desc = "\033[44m TRAINING\033[0m"
        loader = tqdm(self.train_loader, desc=desc)
        write_fn = loader.write

        self.train()
        self.optimizer.zero_grad()
        for index, (images, targets) in enumerate(loader):
            images = images.to(model_device)  # noqa
            targets = targets.to(model_device)

            # At last iteration, the gradient accumulation count can not
            # be equal to gradient accumulation step, so we must perform
            # optimization step when we are at the last iteration (length - 1)
            is_last_index = index >= (length - 1)
            self.train_step(images, targets, write_fn, is_last_index)

            loss_data = {
                "cross_entropy_loss": self.cross_entropy_loss.avg(),
                "precision_score": self.precision_score.avg(),
                "recall_score": self.recall_score.avg(),
                "f1_score": self.f1_score.avg()
            }
            loader.set_postfix(loss_data)

        return loss_data

    def validate(self):
        model_device = self.device()
        loss_data = {}
        self.cross_entropy_loss.reset()
        self.precision_score.reset()
        self.recall_score.reset()
        self.f1_score.reset()

        self.eval()
        with torch.no_grad():
            desc = "\033[43m VALIDATION\033[0m"
            loader = tqdm(self.val_loader, desc=desc)

            for images, targets in loader:
                images = images.to(model_device)  # noqa
                targets = targets.to(model_device)

                # Forward pass
                images = self.input(images)
                logits = self.forward(images)
                predictions, _ = self.output(logits)

                # Comput losses
                loss = self.cross_entropy(logits, targets)

                y_true = targets.cpu().detach().numpy()
                y_pred = predictions.cpu().detach().numpy()

                # Compute metrics
                args = dict(
                    y_true=y_true, y_pred=y_pred, average='macro',
                    zero_division=0
                )
                precision = m.precision_score(**args)
                recall = m.recall_score(**args)
                f1 = m.f1_score(**args)

                self.cross_entropy_loss += loss.item()
                self.precision_score += precision
                self.recall_score += recall
                self.f1_score += f1

                loss_data.update(
                    {
                        "cross_entropy_loss": self.cross_entropy_loss.avg(),
                        "precision_score": self.precision_score.avg(),
                        "recall_score": self.recall_score.avg(),
                        "f1_score": self.f1_score.avg()
                    }
                )
                loader.set_postfix(loss_data)

        return loss_data

    def fit(self):
        """
        Training process
        ----------------

        Run the training loop and returns the results
        formatted as dictionary

        :rtype: `dict`
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.load_checkpoint()

        if not self.metric:
            self.metric = Metric(self.num_epochs, 2)
            self.metric.channels[0] = "train"
            self.metric.channels[1] = "val"

        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            logger.info(f'Epoch: {epoch + 1} / {self.num_epochs}:')

            train_losses = self.train_one_epoch()

            # Update the learning rate
            self.lr_scheduler.step()

            # Add losses to train losses epochs
            # Save checkpoint with the current model state
            # self._add_to_epoch_results(self.train_losses, train_losses)
            self.metric[epoch] = {
                name: {"train": value} for name, value in train_losses.items()
            }

            logger.info(f'{self.print_results(train_losses)}')

            # Make checkpoint after training
            self.checkpoint()

            val_losses = self.validate()

            # self.add_to_epoch_results(self.val_losses, val_losses)
            self.metric[epoch] = {
                name: {"val": value} for name, value in val_losses.items()
            }

            logger.info(f'{self.print_results(val_losses)}')

            if epoch != (self.num_epochs - 1):
                # Epochs are remaining
                logger.info(("-" * 80) + "\n")
                time.sleep(5)
                clear_console()

        self.save("saved_model")
        logger.info(
            "Fine-tuning model is saved at\033[92m saved_model\033[0m."
        )


def parse_argument():
    """
    Command line argument parsing

    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(prog="ALEX-NET Training")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-dt', '--train-data-dir', type=str, required=True)
    parser.add_argument('-dv', '--val-data-dir', type=str, required=True)
    parser.add_argument('-b', '--batch-size', type=int, default=1)

    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--img-channels', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('-nc', '--num-classes', type=int, default=10)

    parser.add_argument('-n', '--epochs', type=int, default=2)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('-gas', type=int, default=128)

    parser.add_argument('-r', "--resume", type=str)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('-m', '--model-file', type=str, help="Alex-Net model")
    parser.add_argument('--best-model', type=str, default="best")

    parser.add_argument(
        '--freeze-feature-layers', action="store_true",
        help="Fine tuning model: Freeze feature layer--require_grad=True"
    )

    args = parser.parse_args()
    logger.info("Training arguments:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    return args


def main():
    """
    Main function to run train process
    """
    args = parse_argument()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_file = args.model_file
    checkpoint = args.resume
    model = None
    if checkpoint:
        model = Training.load(checkpoint_file=checkpoint)
    if not model and model_file:
        model = Training.load(model_file)
        model, optimizer = fine_tune_model(
            model, args.num_classes, args.learning_rate,
            args.freeze_feature_layers, args.weight_decay
        )
        model.optimizer = optimizer
    if not model:
        config = ModelConfig()
        config.img_channels = args.img_channels
        config.img_size = [args.img_size, args.img_size]
        config.num_classes = args.num_classes
        config.dropout_prob = args.dropout

        model = Training(config)

    model = model.to(device)

    model.compile(args)
    model.summary(args.batch_size)
    model.fit()


if __name__ == '__main__':
    try:
        main()
        exit(0)
    except KeyboardInterrupt:
        print("\033[91mCanceled by user!\033[0m")
        exit(125)
