# attention interpretations baseline

---

## LTSM baseline
Libraries Used:
https://github.com/Currie32/Movie-Reviews-Sentiment <br />
Dataset used:
https://www.kaggle.com/c/word2vec-nlp-tutorial/data <br />
Description:
Used for training LTSM on Kaggle movie reviews dataset of 50,000 reviews for sentiment analysis, and use gradient measure for input attribution

---

## Attention baseline using Keras
Libraries Used:
https://github.com/thushv89/attention_keras <br />
Dataset used:
https://github.com/udacity/deep-learning/tree/master/language-translation/data <br />
Description:
Used for training Attention Model for machine translation and create heatmaps using 
1. Energy of attention_states
2. Leave one out
3. Random weight permutation
4. Integrated Gradients<br />

---

## Integrated Gradients
For recurrent neural networks for LSTM (and Attention model) <br /> 


## To Do
1. Add random zero padding to inputs 
2. Add distraction cell to Attention baseline
3. Add distracgtion cell to LSTM baseline
4. Implement IG on attention model
5. Add more Datasets for LSTM and Attention model
6. Leave One out and random permutation methods for attributions for attention model<br />


