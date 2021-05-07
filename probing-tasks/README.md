# Edge Probing Experiments
We used the Edge Probing part of the [Jiant Probing Suite](https://github.com/nyu-mll/jiant-v1-legacy) by Wang et al. to probe the abilities of BERT's layers. In addition to the included tasks of NEL, COREF and REL we added the tasks of Question Type Classification (QUES) and Supporting Facts Extraction (SUP). The code in this section consists of classes for converting datasets into these tasks in the Jiant Edge Probing format.

The base classes in `task_processors.py` can be extended to add even more Probing Tasks.

## Getting Started
The code runs on Python >= 3.6.1.

Install requirements:

`pip install -r requirements.txt`

Run dataset processor with arguments. 
E.g. for converting the SQuAD dataset into the Edge Probing Task 'Supporting Facts Extraction':

```shell
python squad_sup_facts_processor.py \
    -i {path to SQuAD v1.1 dataset} \
    -o {output path for edge probing files, defaults to "./output"}
```
