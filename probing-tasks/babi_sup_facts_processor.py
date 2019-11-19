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
from task_processors import JiantSupportingFactsProcessor


class BABISupportingFactsProcessor(JiantSupportingFactsProcessor):
    DOC_ID = "babi_sup_facts"

    def process_file(self) -> List:
        samples = []

        question_id = 0
        with open(self.input_path, encoding="latin-1") as input_file:
            lines = input_file.readlines()

            current_context = {}  # filled with all sentences belonging to one question

            for line in lines:
                line = self.separate_dots(line)

                if line.startswith("1 "):
                    current_context = {}  # if sentence counter is reset to 1, a new context begins

                sent_key_text_split = line.split(" ", 1)
                sent_key = sent_key_text_split[0]
                text = sent_key_text_split[1]

                tab_split = text.split("\t")

                if len(tab_split) > 2:  # lines containing a question have an additional tab for supporting facts ids
                    question = self.separate_question_mark(text)
                    question_token_length = len(question.split(" "))  # count question tokens

                    sup_facts = tab_split[2].split(" ")
                    sup_facts = [int(s) for s in sup_facts]  # store supporting fact ids

                    sent_keys = sorted(current_context)

                    targets = []
                    context = ""
                    current_token_index = question_token_length

                    # go through all sentences in current context
                    for sent_key in sent_keys:
                        sent = current_context[sent_key]
                        token_count = len(sent.split(" "))

                        sentence_span = [current_token_index, current_token_index + token_count]
                        current_token_index = current_token_index + token_count

                        if sent_key in sup_facts:
                            # mark sentences that belong to supporting facts with label "1"
                            targets.append(self.create_target(question_token_length, sentence_span, "1"))
                        else:
                            # mark sentences that do not belong to supporting facts with label "0"
                            targets.append(self.create_target(question_token_length, sentence_span, "0"))

                        context += sent + " "

                    entry = {"info": {"doc_id": self.DOC_ID,
                                      "q_id": str(question_id)},
                             "text": question + " " + context.strip(),
                             "targets": targets}

                    samples.append(entry)
                    question_id += 1

                else:
                    current_context[int(sent_key)] = text  # add sentence to current context dict

        return samples

    @staticmethod
    def separate_dots(line: str) -> str:
        """
        Insert a space before each dot.
        """
        dot_index = line.find(".")
        if dot_index != -1:
            line = line[:dot_index] + " ."
        return line

    @staticmethod
    def separate_question_mark(line: str) -> str:
        """
        Insert a space before each question mark.
        """
        dot_index = line.find("?")
        if dot_index != -1:
            line = line[:dot_index] + " ?"
        return line


if __name__ == '__main__':
    input_path = ""
    output_dir = ""

    processor = BABISupportingFactsProcessor(input_path=input_path, output_dir=output_dir)
    processor.output_task_in_jiant_format()
