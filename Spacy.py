#!/usr/bin/env python
# coding: utf-8

# In[1]:


from spacy.lang.en import English


# In[2]:


nlp = English()


# In[3]:


doc = nlp('Hello World!')


# In[4]:


for token in doc:
    print(token.text)


# In[5]:


doc


# In[6]:


span = doc[1:4]


# In[7]:


span.text


# In[8]:


doc1 = nlp("This framework is spacy, isnt it beautiful!")


# In[9]:


print("Index: ", [token.i for token in doc1])
print("Text: ", [token.text for token in doc1])


# In[10]:


print("is_alpha: ", [token.is_alpha for token in doc1])


# In[11]:


print("is_punct: ", [token.is_punct for token in doc1])


# In[12]:


print("like_num: ", [token.like_num for token in doc1])


# In[13]:


import spacy


# In[14]:


nlp = spacy.load('en_core_web_sm')
# """
# It includes Binary weights, Vocab, Meta Information like Language, pipeline
# """


# In[15]:


doc = nlp('She ate the Pizza')


# In[16]:


for token in doc:
    print(token.text, token.pos_)


# In[17]:


for token in doc:
    print(token.text, token.pos_, token.dep_, token.head.text)


# In[18]:


doc = nlp("Apple is looking for buying an Indian startup in $1 Billion")


# In[19]:


for ent in doc.ents:
    print(ent.text, ent.label_)


# In[20]:


spacy.explain('NORP')


# In[21]:


spacy.explain('NNP')


# In[22]:


spacy.explain("dobj")


# In[23]:


spacy.explain('nsubj')


# In[24]:


from spacy.matcher import Matcher


# In[25]:


matcher = Matcher(nlp.vocab)


# In[26]:


pattern = [{'ORTH': 'iPhone'}, {'ORTH': 'X'}]


# In[27]:


matcher.add('IPHONE_PATTERN', None, pattern)


# In[28]:


doc = nlp("New iPhone X release date is leaked")


# In[29]:


matches = matcher(doc)


# In[30]:


matches


# In[31]:


for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)


# In[32]:


pattern = [
    {'IS_DIGIT': True},
    {'LOWER': 'fifa'},
    {'LOWER': 'world'},
    {'LOWER': 'cup'},
    {'IS_PUNCT': True}
]


# In[33]:


doc = nlp("2018 FIFA World Cup: France Won!")


# In[34]:


matcher.add('NEWS_PATTERN', None, pattern)


# In[35]:


matches = matcher(doc)


# In[36]:


for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)


# In[37]:


pattern = [
    {'LEMMA': 'love', 'POS': 'VERB'},
    {'POS': 'NOUN'}
]


# In[38]:


matcher.add('PET_LOVE', None, pattern)


# In[39]:


doc = nlp("I love elephant, then I used to love dogs but now I love cats")
matches = matcher(doc)


# In[40]:


for match_id, start, end in matches:
    matched_span = doc[start:end]
    print(matched_span.text)


# In[41]:


# Data Structures: Vocab, Lexems, and String Store


# In[42]:


coffe_hash = nlp.vocab.strings['coffee']


# In[43]:


coffe_hash


# In[44]:


assert coffe_hash == 3197928453018144401


# In[45]:


doc = nlp('I love coffee')


# In[46]:


lexeme = nlp.vocab['coffee']


# In[47]:


lexeme.text, lexeme.orth, lexeme.is_alpha, lexeme.is_currency


# In[48]:


nlp = English()


# In[49]:


from spacy.tokens import Doc


# In[50]:


words = ["Hello", "World", "!"]


# In[51]:


spaces = [True, False, False]


# In[52]:


doc = Doc(nlp.vocab, words=words, spaces=spaces)


# In[53]:


doc


# In[54]:


from spacy.tokens import Span


# In[55]:


span = Span(doc, 0, 2)


# In[56]:


span


# In[57]:


span_with_label = Span(doc, 0, 2, label="Greet!")


# In[58]:


span_with_label.label_


# In[59]:


span_with_label.text


# In[60]:


span_with_label


# In[61]:


nlp = spacy.load('en_core_web_lg')


# In[62]:


doc1 = nlp("Hello World!")
doc2 = nlp("Hey Everyone!")


# In[63]:


doc1.similarity(doc2)


# In[64]:


doc = nlp("I like Pizza")
token = nlp("Soap")[0]


# In[65]:


print(doc.similarity(token))


# In[66]:


span = nlp("I like pizza and pasta")[2:3]
doc = nlp("McDonals sells burgers")[-1]


# In[67]:


print(span.similarity(doc))


# In[68]:


class1a = nlp("Man")
class2a = nlp("King")

class1b = nlp("Woman")
class2b = nlp("Queen") 


# In[69]:


print("Man - Woman = ", class1a.similarity(class1b))
print("King - Queen = ", class2a.similarity(class2b))


# In[70]:


doc = nlp("I have a mango!")


# In[71]:


len(doc.vector)


# In[72]:


doc.vector[:10]


# In[73]:


doc[3].vector


# In[74]:


doc1 = nlp("I like cats")
doc2 = nlp("I hate cats")


# In[75]:


doc1.similarity(doc2)


# In[76]:


matcher = Matcher(nlp.vocab)
matcher.add('DOG', None, [{'LOWER': 'golden'}, {'LOWER': 'retriever'}])
doc = nlp("I have a golden retriever")

for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print('Matched span:', span.text)
    print('Root token:', span.root.text)
    print('Root head token:', span.root.head.text)
    print('Previous token:', doc[start - 1].text, doc[start - 1].pos_)


# In[77]:


from spacy.matcher import PhraseMatcher


# In[78]:


matcher = PhraseMatcher(nlp.vocab)


# In[79]:


pattern = nlp("Golden Retriever")
matcher.add('DOG', None, pattern)


# In[80]:


doc = nlp("I have a Golden Retriever")


# In[81]:


for match_id, start, end in matcher(doc):
    span = doc[start:end]
    print('Matched Span: ', span.text)
    print('Start:', start)
    print('End:', end)


# In[82]:


nlp.pipe_names


# In[83]:


nlp.pipeline


# In[84]:


def custom_component(doc):
    print('Doc Length:', len(doc))
    return doc


# In[85]:


nlp.add_pipe(custom_component, first=True)


# In[86]:


print('Pipeline:', nlp.pipe_names)


# In[87]:


from spacy.tokens import Doc, Token, Span


# In[88]:


Doc.set_extension('title', default=None)
Token.set_extension('is_color', default=False)
Span.set_extension('has_color', default=False)


# In[89]:


doc._.title = 'My Document'
token._.is_color = True
span._.has_color = False


# In[90]:


doc = nlp("The sky is blue.")


# In[91]:


doc[3]._.is_color = True


# In[92]:


doc[3]._.is_color 


# In[93]:


from spacy.tokens import Token


# In[94]:


def get_is_color(token):
    colors = ['red', 'green', 'blue']
    return token.text  in colors


# In[95]:


Token.set_extension('is_color', force=True, getter=get_is_color)


# In[96]:


doc = nlp("The sky is blue.")


# In[97]:


print(doc[3]._.is_color, '-', doc[3].text)


# In[98]:


from spacy.tokens import Span


# In[99]:


def get_has_color(span):
    colors = ['red', 'yellow', 'blue']
    return any(token.text in colors for token in span)


# In[100]:


Span.set_extension('has_color', force=True, getter=get_has_color)


# In[101]:


doc = nlp("The sky is blue.")


# In[102]:


print(doc[1:4]._.has_color, '-', doc[1:4].text)
print(doc[0:2]._.has_color, '-', doc[0:2].text)


# In[103]:


from spacy.tokens import Doc


# In[104]:


def has_token(doc, token_text):
    in_doc = token_text in [token.text for token in doc]
    return in_doc


# In[105]:


Doc.set_extension('has_token', force=True, method=has_token)


# In[106]:


doc = nlp("The sky is blue")


# In[107]:


print(doc._.has_token('blue'), '- blue')
print(doc._.has_token('cloud'), '- cloud')


# In[108]:


LOTS_OF_TEXTS = """I walked through the door with you, the air was cold,
But something 'bout it felt like home somehow and I
Left my scarf there at your sister's house,
And you still got it in your drawer even now.
Oh, your sweet disposition and my wide-eyed gaze.
We're singing in the car, getting lost upstate.
The Autumn leaves falling down like pieces into place,
And I can picture it after all these days.
And I know it's long gone,
And that magic's not here no more,
And I might be okay,
But I'm not fine at all.
'Cause there we are again on that little town street.
You almost ran the red 'cause you were looking over me.
Wind in my hair, I was there, I remember it all too well.
Photo album on the counter, your cheeks were turning red.
You used to be a little kid with glasses in a twin-size bed
And your mother's telling stories about you on a tee ball team
You tell me 'bout your past, thinking your future was me.
And I know it's long gone
And there was nothing else I could do
And I forget about you long enough
To forget why I needed to
'Cause there we are again in the middle of the night.
We dance around the kitchen in the refrigerator light
Down the stairs, I was there, I remember it all too well, yeah.
Maybe we got lost in translation, maybe I asked for too much,
And maybe this thing was a masterpiece 'til you tore it all up.
Running scared, I was there, I remember it all too well.
Hey, you call me up again just to break me like a promise.
So casually cruel in the name of being honest.
I'm a crumpled up piece of paper lying here
'Cause I remember it all, all, all too well.
Time won't fly, it's like I'm paralyzed by it
I'd like to be my old self again, but I'm still trying to find it
After plaid shirt days and nights when you made me your own
Now you mail back my things and I walk home alone
But you keep my old scarf from that very first week
'Cause it reminds you of innocence and it smells like me
You can't get rid of it, 'cause you remember it all too well, yeah
'Cause there we are again, when I loved you so
Back before you lost the one real thing you've ever known
It was rare, I was there, I remember it all too well
Wind in my hair, you were there, you remember it all
Down the stairs, you were there, you remember it all
It was rare, I was there, I remember it all too well"""


# In[109]:


# bad way
# docs = [nlp(text) for text in LOTS_OF_TEXTS]


# In[110]:


# perfect way
# docs = list(nlp.pipe(LOTS_OF_TEXTS))


# In[111]:


docs[:10]


# In[ ]:


data = [
    ('This is a text', {'id': 1, 'page_number': 15}),
    ('Add another text', {'id': 2, 'page_number': 16})
]


# In[ ]:


for doc, context in nlp.pipe(data, as_tuples=True):
    print(doc.text, context['page_number'])


# In[ ]:


Doc.set_extension('id', default=None)
Doc.set_extension('page_number', default=None)


# In[ ]:


for doc, context in nlp.pipe(data, as_tuples=True):
    doc._.id = context['id']
    doc._.page_number = context['page_number']


# In[ ]:


doc = nlp.make_doc("Hello World")


# In[ ]:


# Computationally expensive way
doc = nlp("Hello World")


# In[ ]:


# Computationally easier, sometimes a better way
doc = nlp.make_doc("Hello World")


# In[116]:


with nlp.disable_pipes('tagger', 'parser'):
       doc = nlp('Hello World, I am Aditya. I have an idea which Neuralink Ltd. is executing.')
       print([token.pos_ for token in doc])


# In[117]:


doc = nlp('Hello World, I am Aditya. I have an idea which Neuralink Ltd. is executing.')
print([token.pos_ for token in doc])


# In[ ]:




