#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os


# In[2]:


processed_dir = '/data/project/jeewon/coursework/2022-2/nlp/scripts/data/processed'


# In[8]:


poem_train = pd.read_csv(os.path.join(processed_dir, 'poem_train.csv'))
#poem_test = pd.read_csv(os.path.join(processed_dir, 'poem_test.csv'))
reddit_train = pd.read_csv(os.path.join(processed_dir, 'reddit_train.csv'))
#reddit_test = pd.read_csv(os.path.join(processed_dir, 'reddit_test.csv'))
#entire_corpus = pd.read_csv(os.path.join(processed_dir, 'entire_train_corpus.csv'))


# In[9]:


print(poem_train.shape)
#print(poem_test.shape)
print(reddit_train.shape)
#print(reddit_test.shape)


# In[15]:


poem_train_lowercase = [x.lower() for x in poem_train.text.values]
reddit_train_lowercase = [x.lower() for x in reddit_train.text.values]


# In[18]:


with open(os.path.join(processed_dir, 'small_poem_train_corpus.txt'),'r') as f:
    current_line = f.readline().strip()
    if current_line not in poem_train_lowercase:
        raise ValueError


# In[19]:


with open(os.path.join(processed_dir, 'small_reddit_train_corpus.txt'),'r') as f:
    current_line = f.readline().strip()
    if current_line not in reddit_train_lowercase:
        raise ValueError


# In[ ]:




