# attention interpretations baseline

---

## LTSM baseline
Libraries Used:
https://github.com/Currie32/Movie-Reviews-Sentiment
Dataset used:
https://www.kaggle.com/c/word2vec-nlp-tutorial/data
Description:
Used for training LTSM on Kaggle movie reviews dataset of 50,000 reviews for sentiment analysis, and use gradient measure for input attribution

---

## Attention baseline using Keras
Libraries Used:
https://github.com/thushv89/attention_keras
Dataset used:
https://github.com/udacity/deep-learning/tree/master/language-translation/data
Description:
Used for training Attention Model for machine translation and create heatmaps using 
1. Energy of attention_states
2. Leave one out
3. Random weight permutation
4. Integrated Gradients

---

## Integrated Gradients
For recurrent neural networks for LSTM (and Attention model) 


## To Do
1. Implement IG on attention model
2. Add more Datasets for LSTM and Attention model
3. Leave One out and random permutation methods for attributions for attention model


