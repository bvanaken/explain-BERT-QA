"""
Processor to transform the Question Classification task by Li and Roth (2002) into a Jiant Probing Task.
The Question Type Probing Task takes as input a question. The task is to classify the question type into one of 500
fine-grained types, e.g. entity:animal.

Example question in input format:

ENTY:animal What was the first domesticated bird ?

Example probing task result in Jiant format (JSON):

{"info": {"doc_id": "trec-qt", "q_id": 0},
 "text": "What was the first domesticated bird",
 "targets": [{"span1": [0, 6], "label": "ENTY:animal"}]}

"""

from typing import List
import argparse
from task_processors import JiantTaskProcessor


class TRECQuestionTypeProcessor(JiantTaskProcessor):
    DOC_ID = "trec-qt"

    def process_file(self) -> List:
        """
        Converts the TREC-10 Question Classification file into samples for the Question Type Probing task in Jiant format
        :return: A list of samples in jiant edge probing format.
        """
        samples = []

        sample_id = 0
        with open(self.input_path, encoding="latin-1") as input_file:
            lines = input_file.readlines()
            for line in lines:
                line = line[:-1]  # remove line break character
                split_by_first_space = line.split(" ", 1)
                label = split_by_first_space[0]
                text = split_by_first_space[1]

                # text is already tokenized, so a simple split by space is sufficient
                word_count = len(text.split(" "))

                # we mark the whole question as the span as we do not have specific 'edges'
                span = [0, word_count]

                sample = {"info": {"doc_id": "trec-qt", "q_id": sample_id},
                          "text": text,
                          "targets": [{"span1": span, "label": label}]}

                samples.append(sample)
                sample_id += 1

        return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="path to input dataset file", required=True)
    parser.add_argument("-o", "--output_dir", help="directory where train/dev/test files shall be stored",
                        default="./output")
    args = parser.parse_args()

    processor = TRECQuestionTypeProcessor(input_path=args.input_path, output_dir=args.output_dir)
    processor.output_task_in_jiant_format()
