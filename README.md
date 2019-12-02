# How Does BERT Answer Questions?
This repository contains source code for the experiments from the [paper](https://dl.acm.org/citation.cfm?id=3358028) presented at CIKM 2019.

Find our interactive demo that visualizes the results on three Question Answering datasets here: https://demo.datexis.com/visbert

## Edge Probing Experiments
For probing the language abilities in BERT's layers, we used the [jiant probing](https://github.com/nyu-mll/jiant/tree/master/probing) framework by Tenney et al.
We added two additional tasks to their suite: Question Type Classification and Supporting Fact Extraction. The code for creating these tasks can be found in the [probing]() directory.

## Visualizing Token Transformations
To train and evaluate BERT QA models we used the [🤗 Transformers framework](https://github.com/huggingface/transformers) by Huggingface. A simple way to visualize how tokens are transformed by a QA transformer model can be found in the [visualization]() directory. We use a single question as input and output the token representations for each layer of the model within a 2D vector space.