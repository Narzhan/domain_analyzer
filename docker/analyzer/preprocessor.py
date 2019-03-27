import json
import os
from datetime import datetime, timedelta

from .exc import PreprocessException, FetchException
from .tools import build_logger

if os.environ["MODE"] == "domain_analyzer":
    import requests
    from keras.preprocessing.sequence import pad_sequences
    from nltk.stem import WordNetLemmatizer, SnowballStemmer
    from tld import get_tld
    import gensim
    import nltk
    nltk.download('wordnet')


class Preprocessor:

    def __init__(self, domain: str, tf_idf, ensamble_tf_idf, lda_dictionary, lda_model, ensamble_lda,
                 tokenizer, ensamble_we, we_model):
        self.domain = domain
        self.logger = build_logger("preprocessor", "/opt/domain_analyzer/logs/")
        self.result_logger = build_logger("results", "/opt/domain_analyzer/logs/")
        self.stemmer = SnowballStemmer('english')
        try:
            self.mode = os.environ["ERROR_MODE"]
        except KeyError:
            self.mode = "relaxed"
        self.tf_idf = tf_idf
        self.ensamble_tf_idf = ensamble_tf_idf
        self.lda_dictionary = lda_dictionary
        self.lda_model = lda_model
        self.ensamble_lda = ensamble_lda
        self.tokenizer = tokenizer
        self.we_model = we_model
        self.ensamble_we = ensamble_we
        # self.we_model = load_model("/opt/domain_analyzer/analyzer/models/we_model.h5")
        # self.graph = graph

    def fetch_data(self):
        params = {"q": self.domain, "textDecorations": False}
        headers = {"Ocp-Apim-Subscription-Key": os.environ["BING_API_KEY"]}
        try:
            response = requests.get("https://api.cognitive.microsoft.com/bing/v7.0/search", headers=headers,
                                    params=params)
        except requests.exceptions.RequestException as e:
            self.logger.warning("Failed to download data for domain {}, {}".format(self.domain, e))
            raise FetchException("Failed to download data for domain {}, {}".format(self.domain, e))
        else:
            if response.ok:
                data = response.json()
                if "_type" in data and data["_type"] == "SearchResponse":
                    if "PERSIST_DATA" in os.environ:
                        self.persist_data(data)
                    return data
                else:
                    self.logger.warning("No data from api, following response: {}".format(data))
                    raise FetchException("No data from api, following response: {}".format(data))

    def dry_run(self):
        data = {'rankingResponse': {'mainline': {'items': [{'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.0'}, 'resultIndex': 0}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.1'}, 'resultIndex': 1}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.2'}, 'resultIndex': 2}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.3'}, 'resultIndex': 3}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.4'}, 'resultIndex': 4}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.5'}, 'resultIndex': 5}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.6'}, 'resultIndex': 6}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.7'}, 'resultIndex': 7}]}}, 'webPages': {'value': [{'snippet': "Search the world's information, including webpages, images, videos and more. Google has many special features to help you find exactly what you're looking for.", 'isNavigational': True, 'about': [{'name': 'Google'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.0', 'displayUrl': 'www.google.cz', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-11T07:46:00.0000000Z', 'language': 'en', 'url': 'http://www.google.cz/', 'deepLinks': [{'snippet': 'Bezplatná služba od Googlu okamžitě překládá slova, věty a webové stránky mezi angličtinou a více než stovkou dalších jazyků.', 'url': 'http://translate.google.cz/', 'name': 'Google Překladač'}, {'snippet': "Google's free service instantly translates words, phrases, and web pages between English and over 100 other languages.", 'url': 'http://translate.google.cz/?hl=en&tab=wT', 'name': 'Translate'}, {'snippet': 'Find local businesses, view maps and get driving directions in Google Maps. When you have eliminated the JavaScript , whatever remains must be an empty page. Enable JavaScript to see Google Maps.', 'url': 'https://www.google.cz/maps', 'name': 'Google Maps'}, {'snippet': 'Google Images. The most comprehensive image search on the web.', 'url': 'https://www.google.cz/imghp?tbm=isch', 'name': 'Google Images'}, {'snippet': 'Rozšířené vyhledávání. Najít články. se všemi slovy', 'url': 'http://scholar.google.cz/', 'name': 'Google Scholar'}, {'snippet': 'Vyhledávejte knihy v úplném znění v nejucelenějším indexu na světě. Vydavatelé O službě Ochrana soukromí Smluvní podmínky Nápověda O službě Ochrana soukromí Smluvní podmínky Nápověda', 'url': 'http://books.google.cz/', 'name': 'Google Books'}], 'name': 'Google'}, {'snippet': "Gmail is email that's intuitive, efficient, and useful. 15 GB of storage, less spam, and mobile access.", 'isNavigational': False, 'about': [{'name': 'Gmail'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.1', 'displayUrl': 'mail.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-09T23:55:00.0000000Z', 'language': 'en', 'url': 'http://mail.google.com/', 'name': 'Gmail'}, {'snippet': 'Find local businesses, view maps and get driving directions in Google Maps.', 'isNavigational': False, 'about': [{'name': 'Google Maps'}, {'name': 'Google Maps'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.2', 'displayUrl': 'https://maps.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-11T13:36:00.0000000Z', 'language': 'en', 'url': 'https://maps.google.com/', 'name': 'Google Maps'}, {'snippet': "Search the world's information, including webpages, images, videos and more. Google has many special features to help you find exactly what you're looking for.", 'isNavigational': False, 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.3', 'displayUrl': 'www.google.com/webhp', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-11T11:24:00.0000000Z', 'language': 'en', 'url': 'http://www.google.com/webhp', 'name': 'Google'}, {'snippet': 'Comprehensive up-to-date news coverage, aggregated from sources all over the world by Google News.', 'isNavigational': False, 'about': [{'name': 'Google News'}, {'name': 'Google News'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.4', 'displayUrl': 'https://news.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-10T09:36:00.0000000Z', 'language': 'en', 'url': 'https://news.google.com/', 'name': 'Google News'}, {'snippet': "Google's free service instantly translates words, phrases, and web pages between English and over 100 other languages.", 'isNavigational': False, 'about': [{'name': 'Google Translate'}, {'name': 'Google Translate'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.5', 'displayUrl': 'translate.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-10T05:20:00.0000000Z', 'language': 'en', 'url': 'http://translate.google.com/', 'name': 'Google Translate'}, {'snippet': 'Sign in - Google Accounts', 'isNavigational': False, 'about': [{'name': 'Google Account'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.6', 'displayUrl': 'https://accounts.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-10T13:57:00.0000000Z', 'language': 'en', 'url': 'https://accounts.google.com/', 'name': 'Sign in - Google Accounts'}, {'snippet': 'Google Images. The most comprehensive image search on the web.', 'isNavigational': False, 'about': [{'name': 'Google Images'}, {'name': 'Google Images'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.7', 'displayUrl': 'https://images.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-11T12:12:00.0000000Z', 'language': 'en', 'url': 'https://images.google.com/', 'name': 'Google Images'}], 'totalEstimatedMatches': 234000000, 'webSearchUrl': 'https://www.bing.com/search?q=google.com'}, 'queryContext': {'originalQuery': 'google.com'}, '_type': 'SearchResponse'}
        return data

    def persist_data(self, data: dict):
        try:
            os.mkdir("/opt/domain_analyzer/data/")
        except Exception:
            pass
        with open("/opt/domain_analyzer/data/{}.json".format(self.domain), "w") as out_file:
            json.dump(data, out_file)

    def lemmatize_stemming(self, text):
        return self.stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

    def text_preprocess(self, text):
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(self.lemmatize_stemming(token))
        return result

    def tfidf_analysis(self, texts: list) -> int:
        features = self.tf_idf.transform(texts)
        features = [features.toarray().flatten()]
        return self.ensamble_tf_idf.predict(features)[0]

    def topics_analysis(self, texts: list) -> int:
        texts = list(map(self.text_preprocess, texts))
        bowed_texts = [self.lda_dictionary.doc2bow(doc) for doc in texts]
        features = [
            [max(doc_topics, key=lambda value: value[1])[0] if len(doc_topics) > 0 else 420 for doc_topics in
             self.lda_model.get_document_topics(bowed_texts)]]
        return self.ensamble_lda.predict(features)[0]

    def we_analysis(self, texts: list) -> int:
        padded_texts = pad_sequences(self.tokenizer.texts_to_sequences(texts), maxlen=134)
        predictions = self.we_model.predict(padded_texts)
        features = [list(map(lambda x: 1 if x > 0.5 else 0, predictions))]
        return self.ensamble_we.predict(features)[0]

    def process_text(self, texts: list) -> list:
        while len(texts) < 10:
            texts.append("")
        text_features = []
        for method in [self.tfidf_analysis, self.topics_analysis, self.we_analysis]:
            try:
                text_features.append(method(texts))
            except Exception as me:
                self.logger.info(
                    "Failed to execute text method {} for domain {} with error {}".format(method.__name__, self.domain,
                                                                                          me))
                if self.mode == "relaxed":
                    text_features.append(2)
                else:
                    raise PreprocessException(
                        "Failed to execute text method {} for domain {} with error {}".format(method.__name__,
                                                                                              self.domain,
                                                                                              me))
        return text_features

    def parse_domain(self, domain: str):
        try:
            domain_tld = get_tld(domain, as_object=True, fix_protocol=True)
        except Exception as de:
            self.logger.info("Failed to parse domain {}, {}".format(self.domain, de))
        else:
            return domain_tld

    def check_freshness(self, date: str) -> bool:
        if datetime.now() - datetime.strptime(date.split("T")[0], "%Y-%m-%d") < timedelta(days=7):
            return True
        else:
            return False

    def process_metadata(self, metadata: dict):
        try:
            pages = len(metadata["value"])
            matches = metadata['totalEstimatedMatches']
        except Exception as e:
            self.logger.info("Failed to get domain information, {}".format(e))
            pages = 0 if "pages" not in locals() else pages
            matches = 0 if "matches" not in locals() else matches
        texts = []
        fresh, part_path = 0, 0
        domain_tld = self.parse_domain(self.domain)
        for page in metadata["value"]:
            try:
                texts.append(page["snippet"])
                url_tld = self.parse_domain(page["url"])
                if url_tld and domain_tld:
                    if domain_tld.fld == url_tld.fld:
                        part_path += 1
                        if "dateLastCrawled" in page and self.check_freshness(page["dateLastCrawled"]):
                            fresh += 1
            except Exception as e:
                self.logger.info("Failed to process subpage of domain {}, {}".format(self.domain, e))
        return [part_path, fresh, pages, matches], texts

    def prepare_data(self) -> list:
        raw_data = self.fetch_data() if "TEST_MODE" not in os.environ else self.dry_run()
        if "webPages" in raw_data:
            processed_metadata, texts = self.process_metadata(raw_data["webPages"])
        else:
            processed_metadata, texts = [0, 0, 0, 0], []
        processed_text = self.process_text(texts)
        processed_metadata.extend(processed_text)
        self.result_logger.info("{} - {}".format(self.domain, processed_metadata))
        return [processed_metadata]

# model.predict([[7, 6, 8, 234000000, 0.0, 0.0, 0.0]])
