"""
Processor to transform the bAbI Question Answering Dataset into a Jiant Probing Task.
The Supporting Facts Probing Task takes as input a question and a sentence from the context. The task is to decide
whether the sentence is part of the Supporting Facts for this question.

Example question in bAbI format:

1 John moved to the bathroom.
2 Mary got the football there.
3 Mary went back to the kitchen.
4 Where is the football? 	kitchen	2 3

Example probing task result in Jiant format (JSON):

{"info": {"doc_id": "babi_sup_facts", "q_id": "0"},
 "text": "Where is the football ? John moved to the bedroom . Mary got the football there . Mary went to the kitchen .",
 "targets":
    [{"span1": [0, 5], "span2": [5, 11], "label": "0"},
     {"span1": [0, 5], "span2": [11, 17], "label": "1"},
     {"span1": [0, 5], "span2": [17, 24], "label": "1"}]}

"""

from typing import List
import argparse
from nltk.tokenize import WordPunctTokenizer
from task_processors import JiantSupportingFactsProcessor


class BABISupportingFactsProcessor(JiantSupportingFactsProcessor):
    DOC_ID = "babi_sup_facts"

    word_tokenizer = WordPunctTokenizer()

    def process_file(self) -> List:
        """
        Converts a bAbI QA dataset file into samples for the Supporting Facts Probing task in Jiant format
        :return: A list of samples in jiant edge probing format.
        """

        samples = []

        question_id = 0
        with open(self.input_path, encoding="latin-1") as input_file:
            lines = input_file.readlines()

            current_context = {}  # filled with all sentences belonging to one question

            for line in lines:

                if line.startswith("1 "):
                    current_context = {}  # if sentence counter is reset to 1, a new context begins

                key_content_split = line.split(" ", 1)
                sentence_key = key_content_split[0]
                content = key_content_split[1]

                tab_split = content.split("\t")

                text = tab_split[0]
                tokenized_text = self.word_tokenizer.tokenize(text)

                if len(tab_split) == 1:  # lines with context sentences do not contain tabs

                    # add sentence to sample context
                    current_context[int(sentence_key)] = tokenized_text

                else:
                    # Lines containing a question have an additional tab for supporting fact ids. A question line
                    # always denotes the end of one sample.

                    question_length = len(tokenized_text)  # count question tokens
                    question = " ".join(tokenized_text)

                    sup_facts = tab_split[2].split(" ")
                    sup_facts = [int(s) for s in sup_facts]  # store supporting fact ids

                    targets = []
                    context = ""
                    current_token_pos = question_length

                    sentence_keys = sorted(current_context)  # sort sentences in current context per key

                    # go through all sentences in current context
                    for key in sentence_keys:
                        sentence_tokens = current_context[key]
                        sentence_length = len(sentence_tokens)

                        context += " ".join(sentence_tokens) + " "

                        sentence_span = [current_token_pos, current_token_pos + sentence_length]

                        # if sentence belongs to supporting facts, set label to "1"
                        if key in sup_facts:
                            label = "1"
                        else:
                            label = "0"

                        targets.append(self.create_target(question_length, sentence_span, label))

                        current_token_pos = current_token_pos + sentence_length  # increment token position

                    entry = {"info": {"doc_id": self.DOC_ID,
                                      "q_id": str(question_id)},
                             "text": question + " " + context.strip(),
                             "targets": targets}

                    samples.append(entry)
                    question_id += 1

        return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="path to input dataset file", required=True)
    parser.add_argument("-o", "--output_dir", help="directory where train/dev/test files shall be stored", default=".")
    args = parser.parse_args()

    processor = BABISupportingFactsProcessor(input_path=args.input_path, output_dir=args.output_dir)
    processor.output_task_in_jiant_format()
