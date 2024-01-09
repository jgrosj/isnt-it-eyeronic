# Isn't it EYEronic: Cognitively enhanced data for sarcasm detection

This code makes use of the sarcasm detection dataset ([Mishra et al., 2016](https://aclanthology.org/P16-1104.pdf)), which includes text sequences that are labeled as either sarcastic or non-sarcastic as well as eye-tracking data in the form of fixation durations per token. The script "preprocessing data" prepares two datasets to train a binary classification model on. The baseline consists of the original sequence twice, separated by the SEP token. This is going to be measured against a version enhanced with cognitive data. Namely, we add the tokens in the order of their fixation duration to the original sequence after a SEP token. The input is processed via a SentenceBERT model. A binary classifier is stacked on top of the embedding and then trained for sarcasm detection.

The results show that the cognitively enhanced model does not outperform the baseline:

| Evaluation Metric      |baseline      |cognitive data model|
|------------------------|--------------|--------------------|
| precision              | **81.71%**   | 80.29%             |
| recall                 | **76.25%**   | 74.65%             |
| f1 score               | **72.24%**   | 69.90%             |
