import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer  # Pretrained unsupervised model, can be trained again if required

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)      # To train the model, no labelling required as it is unsupervised

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized[5:]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)  # Part of speech tagging, tuple with (word, pos)

            chunkGram = r"""Chunk: {<.*>+}
                                   }<VB.?|IN|DT>+{"""

            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)


            chunked.draw()

    except Exception as e:
        print(str(e))

process_content()
