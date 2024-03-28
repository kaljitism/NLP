#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def respond(message):
    return "I can hear you! You said: {}".format(message)


# In[ ]:


respond("Hello")


# In[ ]:


def echobot():
    user = input("USER: ")
    print("BOT: {}\n".format(user))
    return echobot()


# In[ ]:


echobot()

