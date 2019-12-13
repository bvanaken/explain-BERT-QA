"""
Processor to transform the HotpotQA Dataset into a Jiant Probing Task.
The Supporting Facts Probing Task takes as input a question and a sentence from the context. The task is to decide
whether the sentence is part of the Supporting Facts for this question.

Example question in HotpotQA format (JSON):

{"_id": "5a8c7595554299585d9e36b6",
 "answer": "Chief of Protocol",
 "question": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
 "supporting_facts": [["Kiss and Tell (1945 film)", 0], ["Shirley Temple", 0], ["Shirley Temple", 1]],
 "context": [["Kiss and Tell (1945 film)", ["Kiss and Tell is a 1945 American comedy film starring then 17-year-old
                Shirley Temple as Corliss Archer.", " In the film, two teenage girls cause their respective parents
                much concern when they start to become interested in boys.", " The parents' bickering about which girl
                is the worse influence causes more problems than it solves."]],
            ["Shirley Temple", ["Shirley Temple Black (April 23, 1928 \u2013 February 10, 2014) was an American actress,
                singer, dancer, businesswoman, and diplomat who was Hollywood's number one box-office draw as a child
                actress from 1935 to 1938.", " As an adult, she was named United States ambassador to Ghana and to
                Czechoslovakia and also served as Chief of Protocol of the United States."]],
            ["Janet Waldo", ["Janet Marie Waldo (February 4, 1920 \u2013 June 12, 2016) was an American radio and voice
                actress.", " She is best known in animation for voicing Judy Jetson, Nancy in \"Shazzan\", Penelope
                Pitstop, and Josie in \"Josie and the Pussycats\", and on radio as the title character in
                \"Meet Corliss Archer\"."]]],
 "type": "bridge",
 "level": "hard"}

Example probing task result in Jiant format (JSON):
# Added line breaks in text for readability

{"info":
    {"doc_id": "hotpot_sup_facts", "q_id": "5a8c7595554299585d9e36b6"},
     "text": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell ?
              Kiss and Tell ( 1945 film ) Kiss and Tell is a 1945 American comedy film starring then 17 - year - old
              Shirley Temple as Corliss Archer . In the film , two teenage girls cause their respective parents much
              concern when they start to become interested in boys . The parents ' bickering about which girl is the
              worse influence causes more problems than it solves .
              Shirley Temple Shirley Temple Black ( April 23 , 1928 \u2013 February 10 , 2014 ) was an American actress
              , singer , dancer , businesswoman , and diplomat who was Hollywood ' s number one box - office draw as a
              child actress from 1935 to 1938 . As an adult , she was named United States ambassador to Ghana and to
              Czechoslovakia and also served as Chief of Protocol of the United States .
              Janet Waldo Janet Marie Waldo ( February 4 , 1920 \u2013 June 12 , 2016 ) was an American radio and voice
              actress . She is best known in animation for voicing Judy Jetson , Nancy in \" Shazzan \", Penelope
              Pitstop , and Josie in \" Josie and the Pussycats \", and on radio as the title character in \" Meet
              Corliss Archer \".",
       "targets": [{"span1": [0, 19], "span2": [26, 48], "label": "1"},
                  {"span1": [0, 19], "span2": [48, 70], "label": "0"},
                  {"span1": [0, 19], "span2": [70, 88], "label": "0"},
                  {"span1": [0, 19], "span2": [90, 137], "label": "1"},
                  {"span1": [0, 19], "span2": [137, 164], "label": "1"},
                  [...]
                  {"span1": [0, 19], "span2": [291, 332], "label": "0"}]}

"""

from typing import List, Dict
import argparse
from nltk.tokenize import WordPunctTokenizer
from task_processors import JiantSupportingFactsProcessor


class HOTPOTSupportingFactsProcessor(JiantSupportingFactsProcessor):
    DOC_ID = "hotpot_sup_facts"

    word_tokenizer = WordPunctTokenizer()

    def process_file(self) -> List:
        """
        Converts a Hotpot dataset file into samples for the Supporting Facts Probing task in Jiant format
        :return: A list of samples in jiant edge probing format.
        """

        hotpot_data = self.json_from_file(self.input_path)
        samples = []

        for sample in hotpot_data:

            question_id = sample['_id']
            sup_facts = sample["supporting_facts"]

            # convert context from Hotpot's nested list format into a dictionary
            context_dict = self.create_context_dict(sample["context"])

            question = sample['question']
            tokenized_question = self.word_tokenizer.tokenize(question)
            question_length = len(tokenized_question)

            sample_text = " ".join(tokenized_question) + " "
            current_token_pos = question_length

            targets = []

            # iterate over all paragraphs
            for paragraph_title in context_dict:

                paragraph_sentences = context_dict[paragraph_title]

                tokenized_title = self.word_tokenizer.tokenize(paragraph_title)
                title_length = len(tokenized_title)

                current_token_pos += title_length

                sample_text += " ".join(tokenized_title) + " "

                # collect the indices of supporting fact sentences in this paragraph
                sup_fact_indices = []
                for sup_fact in sup_facts:
                    sup_fact_title = sup_fact[0]
                    sentence_index = sup_fact[1]

                    if sup_fact_title == paragraph_title:
                        sup_fact_indices.append(sentence_index)

                # iterate over sentences in paragraph and create jiant probing target for each one
                for sentence_index, sentence in enumerate(paragraph_sentences):

                    tokenized_sentence = self.word_tokenizer.tokenize(sentence)
                    sample_text += " ".join(tokenized_sentence) + " "

                    sentence_length = len(tokenized_sentence)

                    # define sentence token span for jiant target
                    start_span = current_token_pos
                    end_span = current_token_pos + sentence_length
                    sentence_span = [start_span, end_span]

                    # if sentence belongs to supporting facts, set label to "1"
                    if sentence_index in sup_fact_indices:
                        label = "1"
                    else:
                        label = "0"

                    targets.append(self.create_target(question_length, sentence_span, label))

                    current_token_pos = current_token_pos + sentence_length  # increment token position

            sample = {"info": {"doc_id": self.DOC_ID,
                               "q_id": question_id},
                      "text": sample_text.strip(),
                      "targets": targets}

            samples.append(sample)

        return samples

    @staticmethod
    def create_context_dict(context: List) -> Dict:
        """
        Converts HotpotQA's context format into a dictionary.

        :param context: HotpotQA's format for question contexts. Consists of a list of paragraphs. Each paragraph itself
        is a nested list with the paragraph title as first entry and a list of sentences as the second.
        :return: Dictionary of all paragraphs, where the title is the key and the value is the list of sentences in the
        paragraph.
        """
        context_dict = {}

        for context_element in context:
            context_title = context_element[0]
            context_sentences = context_element[1]

            context_dict[context_title] = context_sentences

        return context_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", help="path to input dataset file", required=True)
    parser.add_argument("-o", "--output_dir", help="directory where train/dev/test files shall be stored",
                        default="./output")
    args = parser.parse_args()

    processor = HOTPOTSupportingFactsProcessor(input_path=args.input_path, output_dir=args.output_dir)
    processor.output_task_in_jiant_format()
