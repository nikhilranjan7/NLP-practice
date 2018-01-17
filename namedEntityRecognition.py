import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer  # Pretrained unsupervised model, can be trained again if required


"""
NE Type and Examples

ORGANIZATION - Georgia-Pacific Corp., WHO
PERSON - Eddy Bonte, President Obama
LOCATION - Murray River, Mount Everest
DATE - June, 2008-06-29
TIME - two fifty a m, 1:30 p.m.
MONEY - 175 million Canadian Dollars, GBP 10.40
PERCENT - twenty pct, 18.75 %
FACILITY - Washington Monument, Stonehenge
GPE - South East Asia, Midlothian
"""

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)      # To train the model, no labelling required as it is unsupervised

tokenized = custom_sent_tokenizer.tokenize("Nikhil and John are working at Google.")

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)  # Part of speech tagging, tuple with (word, pos)

            namedEnt = nltk.ne_chunk(tagged)
            multiline_string = nltk.chunk.tree2conllstr(namedEnt)
            print(multiline_string.split()[1::3])
            namedEnt.draw()



    except Exception as e:
        print(str(e))

process_content()
