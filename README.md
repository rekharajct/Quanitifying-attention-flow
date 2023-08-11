# Quanitifying attention flow in Transformers

<!-- ABOUT THE PROJECT -->
## Description

This directory contains the Pytorch implementation of the paper Quantifying Attention Flow in Transformers by Samira Abnanar and Willem Zuidema. We have referred  the codes from the repository https://github.com/samiraabnar/attention_flow  for reproducing the results.     
     
     
## To train the verb-number classifier
```
python bert_sv_train.py
```
## To find the attention matrices and relevance scores
```
python sv_find_attention.py
```
## To get the results of correlation scores

```
python sv_attention_results_sci_raw_sum.py

```

## References
```
@inproceedings{abnar-zuidema-2020-quantifying,
    title = "Quantifying Attention Flow in Transformers",
    author = "Abnar, Samira  and
      Zuidema, Willem",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.acl-main.385",
    doi = "10.18653/v1/2020.acl-main.385",
    pages = "4190--4197",
    
}

```
