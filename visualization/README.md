# Visualizing Token Transformations
The code in this section allows to plot 2D vector representations of each layer of a BERT Question Answering model. The plots are built per sample. An example QA sample can be found in `sample.json`.

## Getting Started
The code runs on Python >= 3.6.1.

1. Install requirements:

`pip install -r requirements.txt`

2. Run visualization with arguments (only `-s` and `-m` are required):

```shell
python hidden_state_visualizer.py \
    -s {QA sample JSON-file} \
    -m {path to BERT model file} \
    --bert_model {optional} \
    --output_dir {plots are saved here, defaults to "./output"} \
    --cache_dir {cache directory} \
    --lower_case {True for lower-cased BERT models} \
    --plot_title {experiment title to use for plots}
```