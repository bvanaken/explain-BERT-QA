import os
import json
import random
from typing import List, Dict


class JiantTaskProcessor:
    """
    Base class for bringing a task dataset into jiant's format.
    """

    def __init__(self, input_path: str, output_dir: str, test_ratio: float = 0.1, dev_ratio: float = 0.15):
        self.input_path: str = input_path
        self.output_dir: str = output_dir
        self.test_ratio: float = test_ratio
        self.dev_ratio: float = dev_ratio

    def process_file(self) -> List:
        raise NotImplementedError

    def output_task_in_jiant_format(self) -> None:
        samples = self.process_file()

        # samples = [entry for entry in samples if len(entry["text"].split(" ")) < 400]

        random.shuffle(samples)

        sample_size = len(samples)

        test_size = int(sample_size * self.test_ratio)
        dev_size = int(sample_size * self.dev_ratio)

        test_samples = samples[:test_size]
        dev_samples = samples[test_size:test_size + dev_size]
        train_samples = samples[test_size + dev_size:]

        self.write_samples_to_file(test_samples, os.path.join(self.output_dir, "test.json"))
        self.write_samples_to_file(dev_samples, os.path.join(self.output_dir, "dev.json"))
        self.write_samples_to_file(train_samples, os.path.join(self.output_dir, "train.json"))

    @staticmethod
    def write_samples_to_file(samples: List, output_path: str) -> None:
        if len(samples) > 0:
            with open(output_path, "w", encoding="utf-8") as output:
                for sample in samples:
                    output.write(json.dumps(sample) + "\n")


class JiantSupportingFactsProcessor(JiantTaskProcessor):

    def process_file(self) -> List:
        raise NotImplementedError

    @staticmethod
    def create_target(question_length: int, sentence_span: List[int], label: str) -> Dict:
        """
        Creates a jiant target where the first span contains the question and the second span contains a sentence.
        The label (1 or 0) indicates whether the sentence belongs to the supporting facts for this question.

        :param question_length: number of tokens in question
        :param sentence_span: integer list containing start and end token index of the sentence
        :param label: 1 if the sentence is part of supporting facts, 0 if not
        :return:
        """
        return {"span1": [0, question_length], "span2": sentence_span, "label": label}
