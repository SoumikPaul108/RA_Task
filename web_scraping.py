#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import newspaper
from newspaper import Article
from newspaper import Config


# In[18]:


#!pip install newspaper3k


# In[2]:


df1=pd.read_csv('HTML.csv')
df1


# In[3]:


#count number of rows
row_count=len(df1)
print(row_count)


# In[4]:


#creating a new dataFrame
df2=pd.DataFrame()
df2['Text']=[]
df2


# In[5]:


#iterating over each rows
j=0
for i in range(len(df1)):
    url=(df1.iloc[i]["HTML_FILES"])
    article = newspaper.Article(url)
    try:
        article.download()
        article.parse()
        article.nlp()
    except:
        pass
    # df2 = df2.append({'Text': article.text}, ignore_index=True)
    df2.loc[j]=article.text
    j+=1
df2


# In[6]:


df2.to_csv('TextFile.csv',index=False)


# Extracting Embeddings

# In[27]:


# pip install simpletransformers


# In[22]:


#converting the dataframe into list
text_list=df2['Text'].astype(str).tolist()
# print(text_list)


# In[28]:


# pip install torch


# In[8]:


import numpy as np
import torch
from tqdm.auto import tqdm
from simpletransformers.language_representation import RepresentationModel


# In[23]:


model=RepresentationModel(
    model_type="bert",
    model_name="bert-base-uncased",
    use_cuda=False
)
text_vectors=model.encode_sentences(text_list,combine_strategy="mean")


# In[24]:


text_vectors.shape


# In[25]:


text_vectors

