from typing import List, Tuple

import argparse
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE

from plotting import Token2DVector, TokenPlotter, TokenLabel
from model_wrapper import BertQAModel
from sample_wrapper import QASample


class QAHiddenStateVisualizer:
    def __init__(self, input_sample: QASample, qa_model: BertQAModel, output_path: str = None, plot_title: str = None):
        self.input_sample = input_sample
        self.qa_model = qa_model
        self.output_path = output_path
        self.plot_title = plot_title

    def run_visualization(self):

        prediction, hidden_states, features = self.qa_model.tokenize_and_predict(self.input_sample)
        tokens: List = features.tokens

        # collect char positions for special tokens
        question_indices: Tuple = self.get_question_indices(tokens)
        prediction_indices: Tuple = (prediction["start_index"], prediction["end_index"])

        # assign each token a label such as QUESTION, SUP_FACT, ...
        token_labels: List[TokenLabel] = self.get_labels_for_tokens(tokens, question_indices, prediction_indices,
                                                                    features.sup_ids)

        # build pca-layer list from hidden states
        for layer_index, layer in enumerate(hidden_states):

            # cut off padding
            token_vectors: List = layer[0][:len(tokens)]

            # dimensionality reduction
            layer_reduced: List = self.reduce(data=token_vectors,
                                              method="pca",
                                              dims=2)

            # build list with point information
            token_vectors = []
            for token_index, value in enumerate(layer_reduced[0]):  # iterate over x-values

                token_vector = Token2DVector(x=value,
                                             y=layer_reduced[1][token_index],
                                             token=tokens[token_index],
                                             label=token_labels[token_index])
                token_vectors.append(token_vector)

            plot_title = "Layer {}".format(layer_index) if self.plot_title is None \
                else "{}: Layer {}".format(self.plot_title, layer_index)

            token_plotter = TokenPlotter(vectors=token_vectors,
                                         title=plot_title,
                                         output_path=self.output_path)
            token_plotter.plot()

    @staticmethod
    def get_labels_for_tokens(tokens, question_pos, prediction_pos, sup_facts_pos):

        token_labels = []

        for token_pos, token in enumerate(tokens):

            if prediction_pos[0] <= token_pos <= prediction_pos[1]:
                token_labels.append(TokenLabel.PREDICTION)
                continue
            if question_pos[0] <= token_pos <= question_pos[1]:
                token_labels.append(TokenLabel.QUESTION)
                continue

            is_supporting_fact_token = False
            for sup_fact_pos in sup_facts_pos:
                if sup_fact_pos[0] <= token_pos <= sup_fact_pos[1]:
                    is_supporting_fact_token = True
                    break

            if is_supporting_fact_token:
                token_labels.append(TokenLabel.SUP_FACT)
            else:
                token_labels.append(TokenLabel.DEFAULT)

        return token_labels

    @staticmethod
    def get_question_indices(tokens: List) -> Tuple:
        """Get start and end position for tokens before the [SEP] token, which are the question tokens in QA."""
        sep_token = '[SEP]'
        start_index = 1  # skip first token as it is [CLS]
        end_index = tokens.index(sep_token) - 1
        return start_index, end_index

    @staticmethod
    def reduce(data: List, method: str, dims: int) -> List:
        """Apply reduction method on current vector list."""
        if method == "pca":
            reduction = PCA(n_components=dims)
        elif method == "ica":
            reduction = FastICA(n_components=dims, random_state=0)
        elif method == "tsne":
            reduction = TSNE(n_components=dims)
        else:
            raise KeyError

        reduced = reduction.fit_transform(data)
        return reduced.transpose()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sample_path", help="directory where sample file is stored", required=True)
    parser.add_argument("-m", "--model_path", help="directory where sample file is stored", required=True)
    parser.add_argument("--bert_model", help="type of bert model / tokenizer used", default="bert-base-uncased")
    parser.add_argument("--output_dir", help="directory to store the output files", default="./output")
    parser.add_argument("--cache_dir", help="directory to store the cache files", default="./cache")
    parser.add_argument("--lower_case", help="whether tokenization was lower or upper case", default=True)
    parser.add_argument("--plot_title", help="title of current experiment")
    args = parser.parse_args()

    sample: QASample = QASample.from_json_file(args.sample_path)

    bert_model: BertQAModel = BertQAModel(
        model_path=args.model_path,
        model_type=args.bert_model,
        lower_case=args.lower_case,
        cache_dir=args.cache_dir)

    visualizer: QAHiddenStateVisualizer = QAHiddenStateVisualizer(input_sample=sample,
                                                                  qa_model=bert_model,
                                                                  output_path=args.output_dir,
                                                                  plot_title=args.plot_title)
    visualizer.run_visualization()


if __name__ == '__main__':
    run()
