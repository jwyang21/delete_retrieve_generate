# Preprocessing

The preprocessing consists of the following steps. 

1. Exclude Non-English datas (w/ langdetect)
2. Add token <end_with_no_punc> (which implies the end of senctence with no punctuation marks)
3. Tokenize each data sample(single sentence).
4. Exclude the data sample if it contains more than k uncommon words. (k : hyperparameter)
