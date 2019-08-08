import re
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D,Flatten,Embedding

from keras.preprocessing.sequence import pad_sequences

path = '/aclImdb/'


def clean_review(text):
    text = re.sub('<[^<]+?>', ' ', text)
    text = text.replace('\\"', '')
    text = text.replace('"', '')
    return text


X_Train = []
Y_Train = []

X_Test = []
Y_Test = []




for filename in os.listdir(Path('aclImdb/train/neg')):
    f = open('aclImdb/train/neg/' + filename, 'r', encoding='utf8')
    X_Train.append(clean_review(f.read()))
    Y_Train.append(0)
    f.close()

for filename in os.listdir(Path('aclImdb/train/pos')):
    f = open('aclImdb/train/pos/' + filename, 'r', encoding='utf8')
    X_Train.append(clean_review(f.read()))
    Y_Train.append(1)
    f.close()

for filename in os.listdir(Path('aclImdb/test/neg')):
    f = open('aclImdb/test/neg/' + filename, 'r', encoding='utf8')
    X_Test.append(clean_review(f.read()))
    Y_Test.append(0)
    f.close()

for filename in os.listdir(Path('aclImdb/test/pos')):
    f = open('aclImdb/test/pos/' + filename, 'r', encoding='utf8')
    X_Test.append(clean_review(f.read()))
    Y_Test.append(1)
    f.close()

X_Train, Y_Train = shuffle(X_Train, Y_Train, random_state=0)
X_Test, Y_Test = shuffle(X_Test, Y_Test, random_state=0)


vectorizer = CountVectorizer(binary=True,
                             stop_words=stopwords.words('english'),
                             lowercase=True,
                             min_df=3,
                             max_df=0.9,
                             max_features=5000)


vectorizer.fit_transform(X_Train)

word2idx = {word: idx for idx, word in enumerate(vectorizer.get_feature_names())}
tokenize = vectorizer.build_tokenizer()
preprocess = vectorizer.build_preprocessor()

def to_sequence(tokenizer,preprocessor,index,text):
    words = tokenizer(preprocessor(text))
    indexes = [index[word] for word in words if word in index]
    return indexes










print(to_sequence(tokenize, preprocess, word2idx,
                  "This is an important test!"))


X_train_sequences = [
    to_sequence(tokenize, preprocess, word2idx, x) for x in X_Train
]
print(X_train_sequences[0])

MAX_SEQ_LENGTH = len(max(X_train_sequences, key=len))
print("MAX_SEQ_LENGTH=", MAX_SEQ_LENGTH)

N_FEATURES = len(vectorizer.get_feature_names())

X_train_sequences = pad_sequences(X_train_sequences,maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)
print(X_train_sequences[0])

model = Sequential()


model.add(Embedding(len(vectorizer.get_feature_names())+1,64,input_length=MAX_SEQ_LENGTH))
model.add(Conv1D(64,5,activation='relu'))
model.add(MaxPooling1D(5))
model.add(Flatten())
model.add(Dense(units=64,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


model.fit(X_train_sequences[:-100],
          Y_Train[:-100],
          epochs=3,
          batch_size=512,
          verbose=1,
          validation_data=(X_train_sequences[-100:], Y_Train[-100:]))


X_test_sequences = [to_sequence(tokenize,preprocess,word2idx, x) for x in X_Test]
X_test_sequences = pad_sequences(X_test_sequences,maxlen=MAX_SEQ_LENGTH, value=N_FEATURES)

scores = model.evaluate(X_test_sequences,Y_Test,verbose=1)

print("Accuracy: ",scores[1]) 
#Accuracy: 0.8696