from gunpowder import *
from gunpowder.ext import torch
from gunpowder.torch import *
import logging


input_shape = Coordinate((32, 128, 128))
fmaps = 32
batch_size = 8
class Vgg3D(torch.nn.Module):

    def __init__(self, input_size, fmaps, downsample_factors=[(2,2,2), (2,2,2), (2,2,2), (2,2,2)]):

        super(Vgg3D, self).__init__()

        current_fmaps = 1
        current_size = Coordinate(input_size)

        features = []
        for i in range(len(downsample_factors)):

            features += [
                torch.nn.Conv3d(
                    current_fmaps,
                    fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm3d(fmaps),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv3d(
                    fmaps,
                    fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm3d(fmaps),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool3d(downsample_factors[i])
            ]

            current_fmaps = fmaps
            fmaps *= 2

            size = current_size / downsample_factors[i]
            """
            assert size * 2 == current_size, \
                "Can not downsample %s by factor of 2" % (current_size,)
            """
            current_size = size

            logging.info(
                "VGG level %d: (%s), %d fmaps",
                i,
                current_size,
                current_fmaps)

        self.features = torch.nn.Sequential(*features)

        classifier = [
            torch.nn.Linear(
                current_size[0] * current_size[1] * current_size[2] * current_fmaps,
                4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                4096,
                4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                4096,
                6)
        ]

        self.classifier = torch.nn.Sequential(*classifier)

        print(self)
    def forward(self, raw):
        shape = tuple(raw.shape)            #changed from size() to shape
        raw_with_channels = raw.reshape(
            shape[0],
            1,
            shape[1],
            shape[2],
            shape[3])

        f = self.features(raw_with_channels)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)

        return y
