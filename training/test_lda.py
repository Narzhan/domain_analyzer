# from sklearn import model_selection, preprocessing, decomposition
# from sklearn.feature_extraction.text import CountVectorizer
import pandas
import numpy as np
import gensim
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(7)
import nltk
nltk.download('wordnet')

dataset = pandas.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
dataset = dataset.replace(np.nan, '', regex=True)
stemmer = SnowballStemmer('english')


def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result


processed_docs = dataset['text'].map(preprocess)

dictionary = gensim.corpora.Dictionary(processed_docs)
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

dictionary.filter_extremes(no_below=0.15, no_above=0.8, keep_n=100000)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
processed_docs=None
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=2)
for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))
for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1 * tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
unseen_document = "The mobile web version is similar to the mobile app. Stay on Amazon.com for access to all the features of the main <b>Amazon</b> website. Previous page . Next page. Back to School essentials. Shop all. Deal of the Day. $139.99 $ 139. 99. See more deals. Back to School picks. Favorites from parents, teachers &amp; teens. Online shopping from a great selection at Books Store.Amazon.com</b> je jedna z nejpopulárnějších internetových značek na světě. Popularitou se může zařadit i ke značkám jako jsou ebay, AOL a v menší míře například i Google. Amazon.com(nebo Amazon.co.uk) je symbolem rozvoje internetu a konkrétně nakupování na internetu.Descubre y compra online: electrónica, moda, hogar, libros, deporte y mucho más a precios bajos en Amazon.es. Envío gratis con <b>Amazon</b> Premium."
bow_vector = dictionary.doc2bow(preprocess(unseen_document))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

# train_x, valid_x, train_y, valid_y = model_selection.train_test_split(dataset['text'], dataset['label'])
#
# encoder = preprocessing.LabelEncoder()
# train_y = encoder.fit_transform(train_y)
# valid_y = encoder.fit_transform(valid_y)
#
# count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# count_vect.fit(dataset['text'])
#
# # transform the training and validation data using count vectorizer object
# xtrain_count = count_vect.transform(train_x)
# xvalid_count = count_vect.transform(valid_x)
#
# lda_model = decomposition.LatentDirichletAllocation(n_components=15, learning_method="online",  n_jobs=-1, batch_size=100)
# X_topics = lda_model.fit_transform(xtrain_count)
# topic_word = lda_model.components_
# vocab = count_vect.get_feature_names()
#
# # view the topic models
# n_top_words = 10
# topic_summaries = []
# for i, topic_dist in enumerate(topic_word):
#     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
#     topic_summaries.append(' '.join(topic_words))
# print(topic_summaries)
