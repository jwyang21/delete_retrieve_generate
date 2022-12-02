#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import random
from random import sample
from langdetect import detect
from collections import Counter
random.seed(2022)
np.random.seed(2022)


# python langdetect package: https://pypi.org/project/langdetect/
# - 이 패키지의 input 문장 영어 여부 분류 결과가 100% 정확하진 않지만, 이 패키지가 영어가 아니라고 판별한 문장들을 다 뺐을 때도 데이터 크기가 충분히 커서 전처리 과정에 포함시킴

# For sampling small trainset, use random.sample()
# - Python’s random module provides a sample() function for random sampling, randomly picking more than one element from the list without repeating elements.            
# - referred to https://pynative.com/python-random-sample/

# In[ ]:


#savedir = '/data/project/jeewon/coursework/2022-2/nlp/data/processed'
#datadir = '/data/project/jeewon/coursework/2022-2/nlp/data/'
datadir = './data/raw'
savedir = './data/processed'
if not os.path.exists(savedir):
    os.makedirs(savedir)


# In[ ]:


numbers = ['0','1','2','3','4','5','6','7','8','9']
marks= ['©','*', '..................','- - - - - - - - - - - - - -', 'C̨̼̱è̵͚̬͖̠̜͡r̨͚̜̖̥̗̥͟͡ͅv̩̼e͉̖̭̙̳̗̱͖ͅl͘҉̗̤̠͖ͅo̥̖͍͍̟', '😂', '🔥','👉', 'v̩̼e͉̖̭̙̳̗̱͖ͅl͘҉̗̤̠͖ͅo̥̖͍͍̟', '🐶','🐕','🐩','🐅','🐆','🐾','🌷','❤','💙','💚','💛','❤']
unavailable_strings = ['copyright',  'published by']  
en_strings = 'a b c d e f g h i j k l m n o p q r s t u v w x y z'.split(' ')
small_trainset_size = 221630
test_size = 500


# In[ ]:


def exclude_unavailables(feature, poem_flag, df): 
    
    print("input df shape: ", df.shape)
    #feature: numbers, marks, unavailable_strings
    
    data_ = 'poem' if poem_flag==True else 'reddit'
    
    globals()[data_+'_'+feature+'_indices'] = []
    
    for i in range(df.shape[0]):
        current_string = str(df.text.values[i]).lower()
        #if detect(current_string) != 'en':
        for k in globals()[feature]:
            if k in current_string and i not in globals()[data_+'_'+feature+'_indices']:
                globals()[data_+'_'+feature+'_indices'].append(i)

    print("num_excluded: ", len(globals()[data_+'_'+feature+'_indices']))
    
    if df.iloc[globals()[data_+'_'+feature+'_indices'],:].shape[0] >= 10:
        print("example sentences that are excluded: ")
        print(df.iloc[globals()[data_+'_'+feature+'_indices'],:].sample(10).text.values)
    else:
        print("example sentences that are excluded: ")
        print(df.iloc[globals()[data_+'_'+feature+'_indices'],:].text.values)

    df.drop(globals()[data_+'_'+feature+'_indices'], axis = 0, inplace = True)
    print("data shape after exclusion: ", df.shape)
    df.index = np.arange(df.shape[0])
    
    return df


# In[ ]:


def train_test_split(df, s_flag, savedir, train_fname, test_fname, small_train_fname=''):
    #print(df.shape)
    test_ind = np.random.randint(low=0, high=df.shape[0], size=test_size)
    train_ind = np.delete(np.arange(df.shape[0]), test_ind)
    #print(len(test_ind))
    #print(len(train_ind))
    if len(test_ind)+len(train_ind) - df.shape[0]!= 0:
        raise ValueError
    #print("train: {}, test: {}".format(len(train_ind), len(test_ind)))
    
    train = df.loc[train_ind].copy()
    test = df.loc[test_ind].copy()
    #train.index = np.arange(train.shape[0])
    #test.index = np.arange(test.shape[0])
    
    print("train set size: ", train.shape)
    print("test set size: ", test.shape)
        
    # save train/test sets
    train.to_csv(os.path.join(savedir, train_fname), index = False)
    test.to_csv(os.path.join(savedir, test_fname), index = False)
    
    # (optional) make small trainset
    if s_flag==True:
        train.index = np.arange(train.shape[0])
        small_train_ind = sample(np.arange(train.shape[0]).tolist(), small_trainset_size)
        small_trainset = train.iloc[small_train_ind,:].copy()
        small_trainset.to_csv(os.path.join(savedir, small_train_fname), index = False)
        
    return train, test, small_trainset


# In[ ]:


# import data


# In[ ]:


poem = pd.read_csv(os.path.join(datadir, 'poem1_15.csv'))


# In[ ]:


reddit = pd.read_csv(os.path.join(datadir, 'reddit_15.csv'))


# In[ ]:


print(poem.shape)
print(reddit.shape)


# ---

# ## Poem

# ### Inspect unavailables in poem and reddit data

# ### 1. exclude not-containing-language sentences

# In[ ]:


no_words_sentence_indices = []
for i in range(poem.shape[0]):
    current_string = str(poem.text.values[i]).lower()
    counter_dictionary = Counter(current_string)
    num_en_str = 0
    for en_str in en_strings:
        if en_str in list(counter_dictionary.keys()):
            num_en_str += counter_dictionary[en_str]
    if num_en_str == 0 and i not in no_words_sentence_indices:
        no_words_sentence_indices.append(i)
print(len(no_words_sentence_indices))
print(poem.iloc[no_words_sentence_indices,:].sample(10).text.values)
poem_exclude_no_words = poem.drop(no_words_sentence_indices, axis=0).copy()
poem_exclude_no_words.index = np.arange(poem_exclude_no_words.shape[0])
print(poem_exclude_no_words.shape)


# ### 2. exclude sentences with num_words <= 3

# In[ ]:


few_words_sentence_indices = []
for i in range(poem_exclude_no_words.shape[0]):
    current_string = str(poem_exclude_no_words.text.values[i]).lower()
    if len(current_string.split(' ')) <= 3 and i not in few_words_sentence_indices:
        few_words_sentence_indices.append(i)
print(len(few_words_sentence_indices))
print(poem_exclude_no_words.iloc[few_words_sentence_indices,:].sample(10).text.values)
poem_exclude_few_words = poem_exclude_no_words.drop(few_words_sentence_indices, axis=0).copy()
poem_exclude_few_words.index = np.arange(poem_exclude_few_words.shape[0])
print(poem_exclude_few_words.shape)


# ### 3. exclude numbers

# In[ ]:


poem_exclude_number = exclude_unavailables(feature='numbers', poem_flag=True, df=poem_exclude_few_words.copy())


# ### 4. exclude marks

# In[ ]:


poem_exclude_mark = exclude_unavailables(feature='marks', poem_flag=True, df=poem_exclude_number.copy())


# ### 5. exclude not-english-sentences

# In[ ]:


not_en_indices = []
for i in range(poem_exclude_mark.shape[0]):
    current_string = str(poem_exclude_mark.text.values[i]).lower()
    if detect(current_string) != 'en' and i not in not_en_indices:
        not_en_indices.append(i)
print(len(not_en_indices))
print(poem_exclude_mark.iloc[not_en_indices,:].sample(10).text.values)
poem_exclude_not_en = poem_exclude_mark.drop(not_en_indices, axis=0).copy()
poem_exclude_not_en.index = np.arange(poem_exclude_not_en.shape[0])
print(poem_exclude_not_en.shape)


# ### 6. exclude 'copyright' and 'published by'

# In[ ]:


poem_exclude_unavailables = exclude_unavailables(feature='unavailable_strings', poem_flag=True, df=poem_exclude_not_en.copy())
# 'copyright'이라는 단어가 실제 시의 문장에 쓰인 단어인 경우도 일부 있지만, copyright이 시에 쓰인 경우와 아닌 경우를 하나하나 inpsect하기 힘들어서 일괄적으로 뺌.


# ## train/test split (poem)

# #### (optional) make small corpus for faster training    

# In[ ]:


poem_final = poem_exclude_unavailables.copy()
poem_train, poem_test, small_poem_trainset = train_test_split(poem_final.copy(), True, savedir, 'poem_train.csv', 'poem_test.csv', 'small_poem_train.csv')


# ## make corpus (text only)
# - Use lower case when saving text into corpus

# In[ ]:


#when re-loading data
#poem_train = pd.read_csv(os.path.join(savedir, 'poem_train.csv')) 


# In[ ]:


f = open(os.path.join(savedir, 'poem_train_corpus.txt'), 'w')#entire trainset
for i in range(poem_train.shape[0]):
    f.write(str(poem_train.text.values[i]).lower())
    if i != poem_train.shape[0]-1:
        f.write("\n")
f.close()


# In[ ]:


f = open(os.path.join(savedir, 'poem_test_corpus.txt'), 'w')
for i in range(poem_test.shape[0]):
    f.write(str(poem_test.text.values[i]).lower())
    if i != poem_test.shape[0]-1:
        f.write("\n")
f.close()


# In[ ]:


f = open(os.path.join(savedir, 'small_poem_train_corpus.txt'), 'w')#small trainset <- used this in implementation
for i in range(small_poem_trainset.shape[0]):
    f.write(str(small_poem_trainset.text.values[i]).lower())
    if i != small_poem_trainset.shape[0]-1:
        f.write("\n")
f.close()


# ---

# # Reddit

# ## 1. exclude not-containing-language sentences

# In[ ]:


no_words_sentence_indices = []
for i in range(reddit.shape[0]):
    current_string = str(reddit.text.values[i]).lower()
    counter_dictionary = Counter(current_string)
    num_en_str = 0
    for en_str in en_strings:
        if en_str in list(counter_dictionary.keys()):
            num_en_str += counter_dictionary[en_str]
    if num_en_str == 0 and i not in no_words_sentence_indices:
        no_words_sentence_indices.append(i)
print(len(no_words_sentence_indices))
print(reddit.iloc[no_words_sentence_indices,:].sample(10).text.values)
reddit_exclude_no_words = reddit.drop(no_words_sentence_indices, axis=0).copy()
reddit_exclude_no_words.index = np.arange(reddit_exclude_no_words.shape[0])
print(reddit_exclude_no_words.shape)


# ## 2. exclude sentences with num_words <= 3

# In[ ]:


few_words_sentence_indices = []
for i in range(reddit_exclude_no_words.shape[0]):
    current_string = str(reddit_exclude_no_words.text.values[i]).lower()
    if len(current_string.split(' ')) <= 3 and i not in few_words_sentence_indices:
        few_words_sentence_indices.append(i)
print(len(few_words_sentence_indices))
print(reddit_exclude_no_words.iloc[few_words_sentence_indices,:].sample(10).text.values)
reddit_exclude_few_words = reddit_exclude_no_words.drop(few_words_sentence_indices, axis=0).copy()
reddit_exclude_few_words.index = np.arange(reddit_exclude_few_words.shape[0])
print(reddit_exclude_few_words.shape)


# ## 3. exclude numbers

# In[ ]:


reddit_exclude_number = exclude_unavailables(feature='numbers', poem_flag=False, df=reddit_exclude_few_words.copy())


# ## 4. exclude marks

# In[ ]:


reddit_exclude_mark = exclude_unavailables(feature='marks', poem_flag=False, df=reddit_exclude_number.copy())


# ## 5. exclude not-english-sentences

# In[ ]:


not_en_indices = []
for i in range(reddit_exclude_mark.shape[0]):
#for i in np.arange(259680-1, reddit_exclude_mark.shape[0]):#debug
    current_string = str(reddit_exclude_mark.text.values[i]).lower()
    if detect(current_string) != 'en' and i not in not_en_indices:
        not_en_indices.append(i)
print(len(not_en_indices))
print(reddit_exclude_mark.iloc[not_en_indices,:].sample(10).text.values)
reddit_exclude_not_en = reddit_exclude_mark.drop(not_en_indices, axis=0).copy()
reddit_exclude_not_en.index = np.arange(reddit_exclude_not_en.shape[0])
print(reddit_exclude_not_en.shape)


# ## 6. exclude 'copyright' and 'published by'

# In[ ]:


reddit_exclude_unavailables = exclude_unavailables(feature='unavailable_strings', poem_flag=False, df=reddit_exclude_not_en.copy())


# ## train/test split (reddit)

# In[ ]:


reddit_final = reddit_exclude_unavailables.copy()
reddit_train, reddit_test, small_reddit_trainset = train_test_split(reddit_final.copy(), True, savedir, 'reddit_train.csv', 'reddit_test.csv', 'small_reddit_train.csv')


# ## make corpus (text only)
# - Use lower case when saving text into corpus

# In[ ]:


#when re-loading data
#reddit_train = pd.read_csv(os.path.join(savedir, 'reddit_train.csv'))


# In[ ]:


f = open(os.path.join(savedir, 'reddit_train_corpus.txt'), 'w')#entire trainset
for i in range(reddit_train.shape[0]):
    f.write(str(reddit_train.text.values[i]).lower())
    if i != reddit_train.shape[0]-1:
        f.write("\n")
f.close()


# In[ ]:


f = open(os.path.join(savedir, 'reddit_test_corpus.txt'), 'w')
for i in range(reddit_test.shape[0]):
    f.write(str(reddit_test.text.values[i]).lower())
    if i != reddit_test.shape[0]-1:
        f.write("\n")
f.close()


# In[ ]:


f = open(os.path.join(savedir, 'small_reddit_train_corpus.txt'), 'w')#small trainset <- used this in implementation
for i in range(small_reddit_trainset.shape[0]):
    f.write(str(small_reddit_trainset.text.values[i]).lower())
    if i != small_reddit_trainset.shape[0]-1:
        f.write("\n")
f.close()


# ## save entire corpus

# In[ ]:


f = open(os.path.join(savedir, 'entire_train_corpus.txt'), 'w')
for i in range(poem_train.shape[0]):
    f.write(str(poem_train.text.values[i]).lower())
    f.write("\n")
for k in range(reddit_train.shape[0]):
    f.write(str(reddit_train.text.values[k]).lower())
    if k != reddit_train.shape[0] -1:
        f.write("\n")
f.close()


# In[ ]:


f = open(os.path.join(savedir, 'entire_small_train_corpus.txt'), 'w')
for i in range(small_poem_trainset.shape[0]):
    f.write(str(small_poem_trainset.text.values[i]).lower())
    f.write("\n")
for k in range(small_reddit_trainset.shape[0]):
    f.write(str(small_reddit_trainset.text.values[k]).lower())
    if k != small_reddit_trainset.shape[0] -1:
        f.write("\n")
f.close()


# In[ ]:




