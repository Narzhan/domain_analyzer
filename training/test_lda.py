# from sklearn import model_selection, preprocessing, decomposition
# from sklearn.feature_extraction.text import CountVectorizer
import pandas
import gensim, pickle
from gensim import corpora, models
from gensim.models import CoherenceModel
# import matplotlib.pyplot as plt
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
# from nltk.stem.porter import *
import numpy as np
# np.random.seed(7)
# import nltk
import gensim
# nltk.download('wordnet')

dataset = pandas.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
# dataset = dataset.replace(np.nan, '', regex=True)
# dataset = dataset.sort_index()

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

# pickle.dump(processed_docs, open("processed_docs.pkl", "wb"))
# processed_docs = pickle.load(open("gensim_lda/processed_docs.pkl", "rb"))

dictionary = gensim.corpora.Dictionary(processed_docs)
dictionary.save("dictionary.dict")

# dictionary = pickle.load(open("gensim_lda/old/dictionary.pkl", "rb"))

# dictionary.filter_extremes(no_below=0.10, no_above=0.8, keep_n=100000)
# pickle.dump(dictionary, open("dictionary.pkl", "wb"))
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
# pickle.dump(bow_corpus, open("bow_corpus.pkl", "wb"))
# processed_docs = None
bow_corpus = pickle.load(open("gensim_lda/bow_corpus.pkl", "rb"))

# import pyLDAvis.gensim

lda_model = gensim.models.LdaMulticore.load("gensim_lda/lda_model_20.pkl")
lsi_model = gensim.models.LsiModel(bow_corpus, num_topics=40, id2word=dictionary)
topics = lsi_model[bow_corpus]
# topics = np.array([[doc_topics[0][0]] for doc_topics in lda_model.get_document_topics(bow_corpus)]) # can't be used because the distribution is not sorted
topics = np.array([[max(doc_topics, key=lambda value: value[1])[0] if len(doc_topics) > 0 else 420] for doc_topics in
                   lda_model.get_document_topics(bow_corpus)])
print(topics)

# pickle.dump(topics, open("gensim_lda/old/topics.pkl", "wb"))
# lsi_model = gensim.models.LsiModel(bow_corpus, num_topics=5, id2word=dictionary, )
# visualisation = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
# pyLDAvis.save_html(visualisation, 'LDA_Visualization.html')

# processed_docs = None
# tfidf = models.TfidfModel(bow_corpus)

# corpus_tfidf = tfidf[bow_corpus]

# coherence_values = []
# for k in range(5, 32):
#     lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=k, id2word=dictionary, passes=2, workers=3)
#     processed_docs = pickle.load(open("processed_docs.pkl", "rb"))
#     coherencemodel = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
#     coherence_values.append(coherencemodel.get_coherence())
#     lda_model, processed_docs, coherencemodel = None, None, None

# for doc_topics in lda_model.get_document_topics(bow_corpus):
#     print(doc_topics[0][0])

# lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=2)
# for idx, topic in lda_model_tfidf.print_topics(-1):
#     print('Topic: {} Word: {}'.format(idx, topic))
# for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
#     print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

# for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1 * tup[1]):
#     print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
# unseen_document = "The mobile web version is similar to the mobile app. Stay on Amazon.com for access to all the features of the main <b>Amazon</b> website. Previous page . Next page. Back to School essentials. Shop all. Deal of the Day. $139.99 $ 139. 99. See more deals. Back to School picks. Favorites from parents, teachers &amp; teens. Online shopping from a great selection at Books Store.Amazon.com</b> je jedna z nejpopulárnějších internetových značek na světě. Popularitou se může zařadit i ke značkám jako jsou ebay, AOL a v menší míře například i Google. Amazon.com(nebo Amazon.co.uk) je symbolem rozvoje internetu a konkrétně nakupování na internetu.Descubre y compra online: electrónica, moda, hogar, libros, deporte y mucho más a precios bajos en Amazon.es. Envío gratis con <b>Amazon</b> Premium."
# bow_vector = dictionary.doc2bow(preprocess(unseen_document))
# # print(lda_model.print_topic(max(lda_vector, key=lambda item: item[1])[0]))
#
# for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
#     print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))

#
# def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
#     """
#         Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     texts : List of input texts
#     limit : Max num of topics
#     """
#     coherence_values = []
#     model_list = []
#     perplexity = []
#     for num_topics in range(start, limit, step):
#         model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
#         model_list.append(model)
#         coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
#         coherence_values.append(coherencemodel.get_coherence())
#         perplexity.append(lda_model.log_perplexity(corpus))
#
#     return model_list, coherence_values, perplexity
#
#
# def plot_topic_coherrence():
#     model_list, coherence_values, perplexity = compute_coherence_values(dictionary=dictionary, corpus=bow_corpus, texts=processed_docs,
#                                                             start=2, limit=40, step=6)
#     limit = 40
#     start = 2
#     step = 6
#     x = range(start, limit, step)
#     plt.plot(x, coherence_values)
#     plt.xlabel("Num Topics") #
#     plt.ylabel("Coherence score") #
#     plt.legend(("coherence_values"), loc='best') #
#     plt.show()
#     x = range(start, limit, step)
#     plt.plot(x, perplexity)
#     plt.xlabel("Num Topics") #
#     plt.ylabel("perplexity score") #
#     plt.legend(("perplexity_values"), loc='best') #
#     plt.show()
#
#
# def coherece_perplexity_score():
#     # Compute Perplexity
#     print('\nPerplexity: ', lda_model.log_perplexity(bow_corpus))  # a measure of how good the model is. lower the better.
#
#     # Compute Coherence Score
#     coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
#     coherence_lda = coherence_model_lda.get_coherence()
#     print('\nCoherence Score: ', coherence_lda)
#
# def pyldavis_view():
#     pass


