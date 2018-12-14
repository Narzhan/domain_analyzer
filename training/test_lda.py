# from sklearn import model_selection, preprocessing, decomposition
# from sklearn.feature_extraction.text import CountVectorizer
import pandas
import numpy as np
import gensim
from gensim import corpora, models
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
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
# processed_docs = None
# tfidf = models.TfidfModel(bow_corpus)
# corpus_tfidf = tfidf[bow_corpus]
from pprint import pprint

# for doc in corpus_tfidf:
#     pprint(doc)
#     break
lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)

for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

# lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=2)
# for idx, topic in lda_model_tfidf.print_topics(-1):
#     print('Topic: {} Word: {}'.format(idx, topic))
# for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
#     print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))

# for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1 * tup[1]):
#     print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
unseen_document = "The mobile web version is similar to the mobile app. Stay on Amazon.com for access to all the features of the main <b>Amazon</b> website. Previous page . Next page. Back to School essentials. Shop all. Deal of the Day. $139.99 $ 139. 99. See more deals. Back to School picks. Favorites from parents, teachers &amp; teens. Online shopping from a great selection at Books Store.Amazon.com</b> je jedna z nejpopulárnějších internetových značek na světě. Popularitou se může zařadit i ke značkám jako jsou ebay, AOL a v menší míře například i Google. Amazon.com(nebo Amazon.co.uk) je symbolem rozvoje internetu a konkrétně nakupování na internetu.Descubre y compra online: electrónica, moda, hogar, libros, deporte y mucho más a precios bajos en Amazon.es. Envío gratis con <b>Amazon</b> Premium."
bow_vector = dictionary.doc2bow(preprocess(unseen_document))
# print(lda_model.print_topic(max(lda_vector, key=lambda item: item[1])[0]))

for index, score in sorted(lda_model[bow_vector], key=lambda tup: -1 * tup[1]):
    print("Score: {}\t Topic: {}".format(score, lda_model.print_topic(index, 5)))


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
        Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    """
    coherence_values = []
    model_list = []
    perplexity = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
        perplexity.append(lda_model.log_perplexity(corpus))

    return model_list, coherence_values, perplexity


def plot_topic_coherrence():
    model_list, coherence_values, perplexity = compute_coherence_values(dictionary=dictionary, corpus=bow_corpus, texts=processed_docs,
                                                            start=2, limit=40, step=6)
    limit = 40
    start = 2
    step = 6
    x = range(start, limit, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Num Topics") #
    plt.ylabel("Coherence score") #
    plt.legend(("coherence_values"), loc='best') #
    plt.show()
    x = range(start, limit, step)
    plt.plot(x, perplexity)
    plt.xlabel("Num Topics") #
    plt.ylabel("perplexity score") #
    plt.legend(("perplexity_values"), loc='best') #
    plt.show()


def coherece_perplexity_score():
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(bow_corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_docs, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

def pyldavis_view():
    pass

coherece_perplexity_score()

# from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
# from sklearn.feature_extraction.text import CountVectorizer
#
# NUM_TOPICS = 10
#
# vectorizer = CountVectorizer(min_df=5, max_df=0.9,
#                              stop_words='english', lowercase=True,
#                              token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
# data_vectorized = vectorizer.fit_transform(dataset.text)
#
# # Build a Latent Dirichlet Allocation Model
# lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
# lda_Z = lda_model.fit_transform(data_vectorized)
# print(lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
#
# # # Build a Non-Negative Matrix Factorization Model
# # nmf_model = NMF(n_components=NUM_TOPICS)
# # nmf_Z = nmf_model.fit_transform(data_vectorized)
# # print(nmf_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
# #
# # # Build a Latent Semantic Indexing Model
# # lsi_model = TruncatedSVD(n_components=NUM_TOPICS)
# # lsi_Z = lsi_model.fit_transform(data_vectorized)
# # print(lsi_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)
#
# # Let's see how the first document in the corpus looks like in different topic spaces
# print(lda_Z[0])
# # print(nmf_Z[0])
# # print(lsi_Z[0])
#
#
# def print_topics(model, vectorizer, top_n=10):
#     for idx, topic in enumerate(model.components_):
#         print("Topic %d:" % (idx))
#         print([(vectorizer.get_feature_names()[i], topic[i])
#                         for i in topic.argsort()[:-top_n - 1:-1]])
#
#
# print("LDA Model:")
# print_topics(lda_model, vectorizer)
# print("=" * 20)
#
# # print("NMF Model:")
# # print_topics(nmf_model, vectorizer)
# # print("=" * 20)
# #
# # print("LSI Model:")
# # print_topics(lsi_model, vectorizer)
# # print("=" * 20)
#
# text = "The economy is working better than ever"
# x = lda_model.transform(vectorizer.transform([text]))[0]
# print(x)