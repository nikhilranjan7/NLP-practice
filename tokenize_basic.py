from nltk.tokenize import sent_tokenize, word_tokenize

example_text = "Hello there, how are you doing today? The weather is great and Python is awesome. The sky is pinkish-blue. You should not eat cardboard."

print(sent_tokenize(example_text))

print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)

for i in sent_tokenize(example_text):
    print(i)
