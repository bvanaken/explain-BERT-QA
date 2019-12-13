from typing import Tuple, List

import os
import logging
import torch
from transformers import BertForQuestionAnswering, BertTokenizer, BertConfig

from data_utils import QASample, SquadExample, QAInputFeatures, RawResult, read_squad_example, \
    convert_qa_example_to_features, parse_prediction

logging.basicConfig(format="%(asctime)-15s %(message)s", level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)


class BertQAModel:
    def __init__(self, model_path: str, model_type: str, lower_case: bool, cache_dir: str, device: str = "cpu"):
        self.model_path = model_path
        self.model_type = model_type
        self.lower_case = lower_case
        self.cache_dir = cache_dir
        self.device = device

        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()

    def load_model(self):
        # Load a pretrained model that has been fine-tuned
        config = BertConfig.from_pretrained(self.model_type, output_hidden_states=True, cache_dir=self.cache_dir)

        pretrained_weights = torch.load(self.model_path, map_location=torch.device(self.device))
        model = BertForQuestionAnswering.from_pretrained(self.model_type,
                                                         state_dict=pretrained_weights,
                                                         config=config,
                                                         cache_dir=self.cache_dir)
        return model

    def load_tokenizer(self):
        return BertTokenizer.from_pretrained(self.model_type, cache_dir=self.cache_dir, do_lower_case=self.lower_case)

    def tokenize_and_predict(self, input_sample: QASample) -> Tuple:
        squad_formatted_sample: SquadExample = read_squad_example(input_sample)

        input_features: QAInputFeatures = self.tokenize(squad_formatted_sample)

        with torch.no_grad():
            inputs = {'input_ids': input_features.input_ids,
                      'attention_mask': input_features.input_mask,
                      'token_type_ids': input_features.segment_ids
                      }

            # Make Prediction
            output: Tuple = self.model(**inputs)  # output format: start_logits, end_logits, hidden_states

            # Parse Prediction
            prediction, hidden_states = self.parse_model_output(output, squad_formatted_sample, input_features)

            logger.info("Predicted Answer: {}".format(prediction["text"]))
            logger.info("Start token: {}, End token: {}".format(prediction["start_index"], prediction["end_index"]))

            return prediction, hidden_states, input_features

    def tokenize(self, input_sample: SquadExample) -> QAInputFeatures:
        features = convert_qa_example_to_features(example=input_sample,
                                                  tokenizer=self.tokenizer,
                                                  max_seq_length=384,
                                                  doc_stride=128,
                                                  max_query_length=64,
                                                  is_training=False)

        features.input_ids = torch.tensor([features.input_ids], dtype=torch.long)
        features.input_mask = torch.tensor([features.input_mask], dtype=torch.long)
        features.segment_ids = torch.tensor([features.segment_ids], dtype=torch.long)
        features.cls_index = torch.tensor([features.cls_index], dtype=torch.long)
        features.p_mask = torch.tensor([features.p_mask], dtype=torch.float)

        return features

    @staticmethod
    def parse_model_output(output: Tuple, sample: SquadExample, features: QAInputFeatures) -> Tuple:
        def to_list(tensor):
            return tensor.detach().cpu().tolist()

        result: RawResult = RawResult(unique_id=1,
                                      start_logits=to_list(output[0][0]),
                                      end_logits=to_list(output[1][0]))

        nbest_predictions: List = parse_prediction(sample, features, result)

        return nbest_predictions[0], output[2]  # top prediction, hidden states
