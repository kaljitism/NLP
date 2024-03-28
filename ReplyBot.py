#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[4]:


def swap_pronoun():
    phrase = input("USER: ")
    if 'I' in phrase:
        return re.sub('I', 'you', phrase)
    if 'my' in phrase:
        return re.sub('my', 'your', phrase)
    else:
        return phrase


# In[6]:


swap_pronoun()


# In[ ]:




