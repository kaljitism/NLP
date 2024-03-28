#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[15]:


def do_you_remember():
    pattern = "Do you remember .*"
    message = input("USER: ") 
    match = re.search(pattern, message)
    if match:
        print("How can I forget {}".format(message[16:]))


# In[18]:


do_you_remember()


# In[ ]:




