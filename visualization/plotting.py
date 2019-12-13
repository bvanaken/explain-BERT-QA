from typing import List

import logging
import matplotlib.pyplot as plt
import os
from enum import Enum

logging.basicConfig(format="%(asctime)-15s %(message)s", level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class TokenLabel(Enum):
    DEFAULT = 0
    QUESTION = 1
    PREDICTION = 2
    SUP_FACT = 3


class Token2DVector:
    def __init__(self, x: int, y: int, token: str, label: TokenLabel):
        self.x = x
        self.y = y
        self.token = token
        self.label = label


class TokenPlotter:
    COLOR_LABEL_MAPPING = {
        TokenLabel.DEFAULT: '0.5',
        TokenLabel.PREDICTION: 'red',
        TokenLabel.QUESTION: 'cyan',
        TokenLabel.SUP_FACT: 'green'
    }

    MARKER_LABEL_MAPPING = {
        TokenLabel.DEFAULT: 'o',
        TokenLabel.PREDICTION: 'd',
        TokenLabel.QUESTION: 'o',
        TokenLabel.SUP_FACT: 'o'
    }

    def __init__(self, vectors: List[Token2DVector], title: str, output_path: str = None):
        self.vectors = vectors
        self.title = title
        self.output_path = output_path

    def plot(self):

        for i, vector in enumerate(self.vectors):

            # skip special tokens
            if vector.token == "[SEP]" or vector.token == "[CLS]":
                continue

            color: str = self.COLOR_LABEL_MAPPING[vector.label]
            marker: str = self.MARKER_LABEL_MAPPING[vector.label]

            plt.scatter(vector.x, vector.y, c=color, marker=marker)

            plt.text(vector.x + 0.1, vector.y + 0.2, vector.token, fontsize=6)

        plt.xlabel("PC 1")
        plt.ylabel("PC 2")
        plt.title(self.title)

        if self.output_path is not None:
            self.save_and_close_plot()
            logger.info("{} saved to {}.".format(self.title, self.output_path))
        else:
            logger.info("No output path specified.")
            plt.show()

        # close plot
        plt.clf()

    def save_and_close_plot(self):
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        file_path = "".join([x.lower() if x.isalnum() else "_" for x in self.title])  # prevent invalid file names

        plt.savefig(os.path.join(self.output_path, file_path) + ".pdf")
