# How Does BERT Answer Questions?
This repository contains source code for the experiments from the [paper](https://dl.acm.org/citation.cfm?id=3358028) presented at CIKM 2019.

Find our interactive demo that visualizes the results on three Question Answering datasets here: https://visbert.demo.datexis.com

## Edge Probing Experiments
For probing the language abilities in BERT's layers, we used the [Jiant Probing Suite](https://github.com/nyu-mll/jiant-v1-legacy) by Wang et al.
We added two additional tasks to their suite: Question Type Classification and Supporting Fact Extraction. The code for creating these tasks can be found in the [probing](https://github.com/bvanaken/explain-BERT-QA/tree/master/probing-tasks) directory.

## Visualizing Token Transformations
To train and evaluate BERT QA models we used the [🤗 Transformers framework](https://github.com/huggingface/transformers) by Huggingface. A simple way to visualize how tokens are transformed by a QA transformer model can be found in the [visualization](https://github.com/bvanaken/explain-BERT-QA/tree/master/visualization) directory. We use a single question as input and output the token representations for each layer of the model within a 2D vector space.

## Cite
When building up on our work, please cite our paper as follows:
```
@article{van_Aken_2019,
   title={How Does BERT Answer Questions?},
   journal={Proceedings of the 28th ACM International Conference on Information and Knowledge Management  - CIKM  ’19},
   publisher={ACM Press},
   author={van Aken, Betty and Winter, Benjamin and Löser, Alexander and Gers, Felix A.},
   year={2019}
}
```
