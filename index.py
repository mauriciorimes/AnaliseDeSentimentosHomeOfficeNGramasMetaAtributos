import pandas as pd

dataset = pd.read_csv("novo_dataset_completo.csv")

dataset.head(10)

dataset["classe"].value_counts()

import string

punct_list = "\"$%&'()*+,-./:;<=>@[\]^_`{|}~?!"
exclist = punct_list + string.digits
# remove punctuations and digits
table = str.maketrans('', '', exclist)

def standardize_tweets(dataframe, attribute):
  dataframe[attribute] = dataframe[attribute].str.lower()
  dataframe[attribute] = dataframe[attribute].str.replace(r"@\S+","user")
  dataframe[attribute] = dataframe[attribute].str.replace(r"http\S+","url")
  dataframe[attribute] = dataframe[attribute].str.replace(r"home office","")
  dataframe[attribute] = dataframe[attribute].str.replace(r"home ofice","")
  dataframe[attribute] = dataframe[attribute].str.replace(r"trabalho remoto","")
  dataframe[attribute] = dataframe[attribute].str.translate(table) #retira pontuação e dígitos

standardize_tweets(dataset,"tweet")

dataset.head()

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

lista_de_stopwords_ptbr = stopwords.words('portuguese')

print(lista_de_stopwords_ptbr)

from math import remainder
from nltk.tokenize import TweetTokenizer
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import CountVectorizer

tweet_tokenizer = TweetTokenizer()

preprocess = make_column_transformer(
    (CountVectorizer(tokenizer = tweet_tokenizer.tokenize, ngram_range=(1,1), binary=True, stop_words=lista_de_stopwords_ptbr),"tweet"),
    remainder="passthrough")

pipeline = Pipeline(
    [
        ("vect", preprocess),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ]
)

from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn import metrics

strat_cv = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

avg_acc = 0.0
avg_f1 = 0.0
i = 1

X = dataset.drop(columns=["classe"])
y = dataset["classe"]

for train_index, test_index in strat_cv.split(X, y):

  print(f"\n--------------------- AVALIAÇÃO {i} ---------------------\n")
  print()

  clone_pipeline = clone(pipeline)

  X_train = X.iloc[train_index]
  y_train = y.iloc[train_index]
  X_test = X.iloc[test_index]
  y_test = y.iloc[test_index]

  clone_pipeline.fit(X_train, y_train)
  y_pred = clone_pipeline.predict(X_test)

  print(metrics.classification_report(y_test,y_pred,digits=3))
  print(pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predito'], margins=True))

  avg_acc += accuracy_score(y_test,y_pred)

  prec,rec,f1,supp = score(y_test, y_pred, average='macro')
  avg_f1 += f1

  i += 1

print("\n---> RESULTADOS DA VALIDAÇÃO CRUZADA COM 10 AVALIAÇÕES:")
print(f"Acurácia: {avg_acc/10.0:.4f}")
print(f"F1-score (macro): {avg_f1/10.0:.4f}")
