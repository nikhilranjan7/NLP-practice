from nltk.stem import WordNetLemmatizer

lematizer = WordNetLemmatizer()

print(lematizer.lemmatize("cats"))
print(lematizer.lemmatize("cacti"))
print(lematizer.lemmatize("geese"))
print(lematizer.lemmatize("rocks"))
print(lematizer.lemmatize("python"))
print(lematizer.lemmatize("pythoning"))

print(lematizer.lemmatize("better"))

# Default parameter to "pos" is "n" for noun, if something is not noun we have to pass the pos type of that word
print(lematizer.lemmatize("better", pos="a"))
print(lematizer.lemmatize("best", pos="a"))

print(lematizer.lemmatize("run"))
print(lematizer.lemmatize("run","v"))
