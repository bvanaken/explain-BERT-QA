"""
Processor to transform the SQuAD v1.1 Dataset into a Jiant Probing Task.
The Supporting Facts Probing Task takes as input a question and a sentence from the context. The task is to decide
whether the sentence is part of the Supporting Facts for this question.
As the SQuAD dataset does not include multi-hop questions, we consider the sentence containing the answer as the only
supporting fact.

Example question in SQuAD format (JSON):

{"title": "University_of_Notre_Dame",
 "paragraphs":
 [{"context": "Architecturally, the school has a Catholic character.
               Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection.",
    "qas": [{"answers":
        [{"answers": [{"answer_start": 381, "text": "a Marian place of prayer and reflection"}],
          "question": "What is the Grotto at Notre Dame?", "id": "5733be284776f41900661181"}]}

Example probing task result in Jiant format (JSON):

{"info": {"doc_id": "squad_sup_facts", "q_id": "5726e985dd62a815002e94db"},
 "text": "What is the Grotto at Notre Dame ?  Architecturally , the school has a Catholic character . Immediately behind
          the basilica is the Grotto , a Marian place of prayer and reflection .",
 "targets":
    [{"span1": [0, 8], "span2": [8, 17], "label": "0"},
     {"span1": [0, 8], "span2": [17, 34], "label": "1"}]}

"""

from typing import List
import argparse
from nltk.tokenize import WordPunctTokenizer
import nltk.data
from task_processors import JiantSupportingFactsProcessor


class SQUADSupportingFactsProcessor(JiantSupportingFactsProcessor):
    DOC_ID = "squad_sup_facts"

    word_tokenizer = WordPunctTokenizer()
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def process_file(self) -> List:
        """
        Converts a SQuAD dataset file into samples for the Supporting Facts Probing task in Jiant format
        :return: A list of samples in jiant edge probing format.
        """
        squad_data = self.json_from_file(self.input_path)['data']
        samples = []

        for article in squad_data:
            pars = article["paragraphs"]
            for par in pars:
                context = par["context"]

                tokenized_context = self.word_tokenizer.tokenize(context)
                sentences = list(self.sentence_tokenizer.tokenize(context.strip()))

                if len(sentences) < 2:  # There must be at least two sentences in the paragraph
                    continue

                for qa in par["qas"]:
                    targets = []
                    answer = qa["answers"][0]
                    question = qa["question"]
                    question_id = qa["id"]

                    tokenized_question = self.word_tokenizer.tokenize(question)
                    question_length = len(tokenized_question)
                    sample_text = " ".join(tokenized_question) + " "

                    answer_char_position = answer["answer_start"]
                    answer_sentence_index = self.get_sentence_index_from_char_position(answer_char_position, sentences)

                    found_answer_sentence_in_context = False

                    # go through all sentences in context
                    for sentence_index, sentence in enumerate(sentences):

                        tokenized_sentence = self.word_tokenizer.tokenize(sentence)
                        sample_text += " ".join(tokenized_sentence) + " "

                        # get token start position for sentence in context
                        sentence_pos = self.find_sentence_position_in_context(tokenized_context, tokenized_sentence)

                        if sentence_pos is None:
                            continue

                        # define sentence token span for jiant target
                        start_index = sentence_pos + question_length
                        end_index = start_index + len(tokenized_sentence)
                        sentence_span = [start_index, end_index]

                        # if sentence contains answer, set label to "1"
                        if sentence_index == answer_sentence_index:
                            label = "1"
                            found_answer_sentence_in_context = True
                        else:
                            label = "0"

                        targets.append(self.create_target(question_length, sentence_span, label))

                    if not found_answer_sentence_in_context:
                        # could not find answer in context, skip this example
                        continue

                    sample = {"info": {"doc_id": self.DOC_ID, "q_id": question_id},
                              "text": sample_text.strip(),
                              "targets": targets}

                    samples.append(sample)

        return samples

    @staticmethod
    def find_sentence_position_in_context(context: List, sentence_tokens: List) -> int:
        """
        Goes through a list of context tokens and tries to find the sentence tokens. If sentence tokens are found, the
        start index is returned.

        :param context: List of tokens in a context document.
        :param sentence_tokens: List of tokens in a sentence, that is supposed to be within the context.
        :return: The start token position of the sentence in the context. If not found returns None.
        """
        for token_index, token in enumerate(context):
            # check if current token equals the first sentence token
            if token == sentence_tokens[0]:

                match = True
                # go through all sentence tokens to see if they match with the following context tokens
                for i in range(1, len(sentence_tokens)):
                    if len(context) > token_index + i and context[token_index + i] == sentence_tokens[i]:
                        continue

                    match = False
                    break
                if match:
                    return token_index

    @staticmethod
    def get_sentence_index_from_char_position(char_pos: int, sentences: List) -> int:
        """
        Gets a list of sentences from a paragraph and returns the index of the sentence that contains a certain
        character position.
        :param char_pos: Character position in paragraph
        :param sentences: List of paragraph sentences
        :return: Index of the sentence that contains the character
        """
        char_count = 0
        for sentence_index, sentence in enumerate(sentences):
            char_count += len(sentence)
            if char_count >= char_pos:
                return sentence_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="path to input dataset file", required=True)
    parser.add_argument("-o", "--output_dir", help="directory where train/dev/test files shall be stored",
                        default="./output")
    args = parser.parse_args()

    processor = SQUADSupportingFactsProcessor(input_path=args.input_path, output_dir=args.output_dir)
    processor.output_task_in_jiant_format()
