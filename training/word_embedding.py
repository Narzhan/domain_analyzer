import pandas, nltk
import numpy as np
from gensim.models import word2vec
from gensim.models import FastText
import gensim, pickle
from nltk import SnowballStemmer, WordNetLemmatizer

dataset = pandas.read_csv("text_test_data.csv", index_col=2, encoding='utf-8', delimiter=";", engine="python")
dataset = dataset.replace(np.nan, '', regex=True)
dataset = dataset.sort_index()
stemmer = SnowballStemmer('english')

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# processed_docs = pickle.load(open("gensim_lda/processed_docs.pkl", "rb"))



feature_size = 100  # Word vector dimensionality
window_context = 20  # Context window size
min_word_count = 10  # Minimum word count
sample = 1e-3  # Downsample setting for frequent words

# w2v_model = word2vec.Word2Vec(processed_docs, size=feature_size,
#                               window=window_context, min_count=min_word_count,
#                               sample=sample, iter=10)
# fast_text = FastText(processed_docs, size=feature_size,
#                               window=window_context, min_count=min_word_count,
#                               sample=sample, iter=10)
# fast_text.save("fast_text.pkl")
# w2v_model = word2vec.Word2Vec.load("gensim_we/w2v_model.pkl")



# similar_words = {search_term: [item[0] for item in w2v_model.wv.most_similar([search_term], topn=5)]
#                   for search_term in ['malware', 'phishing', 'hoax', 'exploit', 'botnet', 'spam']}
# print(similar_words)

def average_word_vectors(words, model, vocabulary, num_features):
    feature_vector = np.zeros((num_features,), dtype="float64")
    nwords = 0.

    for word in words:
        if word in vocabulary:
            nwords = nwords + 1.
            feature_vector = np.add(feature_vector, model[word])

    if nwords:
        feature_vector = np.divide(feature_vector, nwords)

    return feature_vector


def averaged_word_vectorizer(corpus, model, num_features):
    vocabulary = set(model.wv.index2word)
    features = [average_word_vectors(tokenized_sentence, model, vocabulary, num_features)
                for tokenized_sentence in corpus]
    return np.array(features)


# w2v_feature_array = averaged_word_vectorizer(corpus=processed_docs, model=w2v_model,
#                                              num_features=feature_size)
# document_array = pandas.DataFrame(w2v_feature_array)
# pickle.dump(w2v_feature_array, open("w2v_feature_array.pkl", "wb"))
w2v_feature_array = pickle.load(open("w2v_feature_array.pkl", "rb"))
w2v_model = None
from sklearn.cluster import AffinityPropagation, MiniBatchKMeans

# ap = AffinityPropagation()
ap = MiniBatchKMeans()
ap.fit(w2v_feature_array)
cluster_labels = ap.labels_
# pickle.dump(cluster_labels, open("cluster_labels.pkl", "wb"))
dataset.reset_index(drop=True, inplace=True)
cluster_labels = pandas.DataFrame(cluster_labels, columns=['ClusterLabel'])
print(pandas.concat([dataset, cluster_labels], axis=1))
