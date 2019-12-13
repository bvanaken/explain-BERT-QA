from typing import List, Dict
from nltk import tokenize
import json


class QASample:
    def __init__(self, sample_id: str, question: str, answer: str, context: str, sup_ids: List[List[int]] = None):
        self.sample_id = sample_id
        self.question = question
        self.context = context
        self.answer_dict = self.build_answer_dict(answer)
        self.sup_ids = sup_ids if sup_ids is not None else [self.sup_id_from_answer_sentence()]

    def build_answer_dict(self, answer: str) -> Dict:
        answer_start = self.context.lower().find(answer.lower())

        return {
            "text": answer,
            "answer_start": answer_start
        }

    def sup_id_from_answer_sentence(self) -> List[List[int]]:
        """Returns the start and end position of the sentence containing the answer string"""
        if self.answer_dict["answer_start"] != -1 and self.answer_dict["text"] != "":
            return self.get_sentence_span_from_char_position(self.context, self.answer_dict["answer_start"])

    @staticmethod
    def get_sentence_span_from_char_position(context, char_position):
        """Get the character span of the sentence containing the char position."""
        ctx_sentences = tokenize.sent_tokenize(context)

        char_count = 0
        for sentence in ctx_sentences:

            start_id = context.lower().find(sentence.lower())
            char_count += len(sentence)
            if char_count > char_position:
                return [start_id, char_count]

    @staticmethod
    def from_json_file(file_path: str):
        """Creates QASample from JSON file with fields
        'question', 'answer', 'context' and optional 'sample_id', 'sup_ids'"""

        with open(file_path) as sample_file:
            content = json.load(sample_file)
            sample_id = content["sample_id"] if "sample_id" in content else "0"
            sup_ids = content["sup_ids"] if "sup_ids" in content else None

            return QASample(sample_id=sample_id,
                            question=content["question"],
                            answer=content["answer"],
                            context=content["context"],
                            sup_ids=sup_ids)
