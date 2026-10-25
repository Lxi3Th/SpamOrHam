import pandas as pd 
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('combined_dataset.csv')

X = dataset['text']
Y = dataset['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

wordCount = CountVectorizer()

words = wordCount.fit_transform(X_train)

magicForm = wordCount.transform(X_test)

model = MultinomialNB()

trainModel = model.fit(words, Y_train)

predict = model.predict(magicForm)

score = accuracy_score(Y_test, predict)

print(f'{score * 100:.2f} % Accurate')

def spamOrNot(msg): 
        word = wordCount.transform([msg])
        test = model.predict(word)
        print(f'Result is {test[0]}')

print("#-- Spam Detector --#")

msg = 'Congrats!!!! you won a free tablet'

spamOrNot(msg)

em = 'Anthropic is hiring'

spamOrNot(em)