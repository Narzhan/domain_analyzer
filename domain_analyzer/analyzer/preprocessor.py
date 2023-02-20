import json
import os
import random
import re
import requests
import string
from typing import Tuple
from datetime import datetime, timedelta
from math import sqrt
from urllib.parse import urlparse
from .exc import PreprocessException, FetchException, NoDataException
from .tools import build_logger

if os.environ["MODE"] == "domain_analyzer":
    import tensorflow as tf
    from keras.preprocessing.sequence import pad_sequences
    from numpy import array
    from gensim.parsing.preprocessing import remove_stopwords


class Preprocessor:
    charmap: dict = {c: i for i, c in enumerate('$' + string.ascii_lowercase + string.digits + '-_.')}
    max_domains: int = 60
    max_texts: int = 60
    max_text_length: int = 25
    max_domains_similarity: int = 20
    max_domain_length: int = 64
    punctuations: str = r'''!()-—+_•[]{};:'"\,<>=·./?@#$%^&*_~'''
    nn_prob_threshold: float = 0.5
    base_path = "/opt/domain_analyzer/analyzer/models/"

    def __init__(self, domain: str, similarity_model, tokenizer):
        self.domain = domain
        self.logger = build_logger("preprocessor", "/opt/domain_analyzer/logs/")
        self.result_logger = build_logger("results", "/opt/domain_analyzer/logs/")
        self.nn_prob_logger = build_logger("nn_prob", "/opt/domain_analyzer/logs/")
        self.similarity_model = similarity_model
        self.cnn_blackbox = tf.keras.models.load_model("{}domains_blackbox_no_embedding_v2.h5".format(self.base_path))
        self.cnn_texts = tf.keras.models.load_model("{}texts_we_glove_cnn_v3.h5".format(self.base_path))
        self.tokenizer = tokenizer

    def fetch_data(self):
        """
            Get data from Bing search engine
        :return:
            dict, representation of returned data from analysed domain
        """
        data = {"target": "google_search", "query": f"\"{self.domain}\"",
                  "parse": True, "locale": "en-GB", "num_pages": 100,
                  "google_results_language": "en", "geo": "Prague"
                  }
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        try:
            response = requests.post("https://scrape.smartproxy.com/v1/tasks", headers=headers,
                                     json=data, auth=(os.environ["SCRAPER_LOGIN"], os.environ["SCRAPER_PASSWORD"]))
        except requests.exceptions.RequestException as e:
            self.logger.warning("Failed to download data for domain {}, {}".format(self.domain, e))
            raise FetchException("Failed to download data for domain {}, {}".format(self.domain, e))
        else:
            if response.ok:
                data = response.json()
                if os.environ.get("PERSIST_DATA", "true") == "true":
                    self.persist_data(data)
                return data
            else:
                self.logger.warning("Failed to download data for {}, {}: {}".format(self.domain, response.status_code,
                                                                                    response.content))

    def dry_run(self) -> dict:
        """
            Get random data from a dry-run option
        :return:
            dict, random data sample from the preloaded ones
        """
        data = [{'rankingResponse': {'mainline': {'items': [{'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.0'}, 'resultIndex': 0}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.1'}, 'resultIndex': 1}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.2'}, 'resultIndex': 2}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.3'}, 'resultIndex': 3}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.4'}, 'resultIndex': 4}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.5'}, 'resultIndex': 5}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.6'}, 'resultIndex': 6}, {'answerType': 'WebPages', 'value': {'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.7'}, 'resultIndex': 7}]}}, 'webPages': {'value': [{'snippet': "Search the world's information, including webpages, images, videos and more. Google has many special features to help you find exactly what you're looking for.", 'isNavigational': True, 'about': [{'name': 'Google'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.0', 'displayUrl': 'www.google.cz', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-11T07:46:00.0000000Z', 'language': 'en', 'url': 'http://www.google.cz/', 'deepLinks': [{'snippet': 'Bezplatná služba od Googlu okamžitě překládá slova, věty a webové stránky mezi angličtinou a více než stovkou dalších jazyků.', 'url': 'http://translate.google.cz/', 'name': 'Google Překladač'}, {'snippet': "Google's free service instantly translates words, phrases, and web pages between English and over 100 other languages.", 'url': 'http://translate.google.cz/?hl=en&tab=wT', 'name': 'Translate'}, {'snippet': 'Find local businesses, view maps and get driving directions in Google Maps. When you have eliminated the JavaScript , whatever remains must be an empty page. Enable JavaScript to see Google Maps.', 'url': 'https://www.google.cz/maps', 'name': 'Google Maps'}, {'snippet': 'Google Images. The most comprehensive image search on the web.', 'url': 'https://www.google.cz/imghp?tbm=isch', 'name': 'Google Images'}, {'snippet': 'Rozšířené vyhledávání. Najít články. se všemi slovy', 'url': 'http://scholar.google.cz/', 'name': 'Google Scholar'}, {'snippet': 'Vyhledávejte knihy v úplném znění v nejucelenějším indexu na světě. Vydavatelé O službě Ochrana soukromí Smluvní podmínky Nápověda O službě Ochrana soukromí Smluvní podmínky Nápověda', 'url': 'http://books.google.cz/', 'name': 'Google Books'}], 'name': 'Google'}, {'snippet': "Gmail is email that's intuitive, efficient, and useful. 15 GB of storage, less spam, and mobile access.", 'isNavigational': False, 'about': [{'name': 'Gmail'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.1', 'displayUrl': 'mail.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-09T23:55:00.0000000Z', 'language': 'en', 'url': 'http://mail.google.com/', 'name': 'Gmail'}, {'snippet': 'Find local businesses, view maps and get driving directions in Google Maps.', 'isNavigational': False, 'about': [{'name': 'Google Maps'}, {'name': 'Google Maps'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.2', 'displayUrl': 'https://maps.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-11T13:36:00.0000000Z', 'language': 'en', 'url': 'https://maps.google.com/', 'name': 'Google Maps'}, {'snippet': "Search the world's information, including webpages, images, videos and more. Google has many special features to help you find exactly what you're looking for.", 'isNavigational': False, 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.3', 'displayUrl': 'www.google.com/webhp', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-11T11:24:00.0000000Z', 'language': 'en', 'url': 'http://www.google.com/webhp', 'name': 'Google'}, {'snippet': 'Comprehensive up-to-date news coverage, aggregated from sources all over the world by Google News.', 'isNavigational': False, 'about': [{'name': 'Google News'}, {'name': 'Google News'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.4', 'displayUrl': 'https://news.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-10T09:36:00.0000000Z', 'language': 'en', 'url': 'https://news.google.com/', 'name': 'Google News'}, {'snippet': "Google's free service instantly translates words, phrases, and web pages between English and over 100 other languages.", 'isNavigational': False, 'about': [{'name': 'Google Translate'}, {'name': 'Google Translate'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.5', 'displayUrl': 'translate.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-10T05:20:00.0000000Z', 'language': 'en', 'url': 'http://translate.google.com/', 'name': 'Google Translate'}, {'snippet': 'Sign in - Google Accounts', 'isNavigational': False, 'about': [{'name': 'Google Account'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.6', 'displayUrl': 'https://accounts.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-10T13:57:00.0000000Z', 'language': 'en', 'url': 'https://accounts.google.com/', 'name': 'Sign in - Google Accounts'}, {'snippet': 'Google Images. The most comprehensive image search on the web.', 'isNavigational': False, 'about': [{'name': 'Google Images'}, {'name': 'Google Images'}], 'id': 'https://api.cognitive.microsoft.com/api/v7/#WebPages.7', 'displayUrl': 'https://images.google.com', 'isFamilyFriendly': True, 'dateLastCrawled': '2019-03-11T12:12:00.0000000Z', 'language': 'en', 'url': 'https://images.google.com/', 'name': 'Google Images'}], 'totalEstimatedMatches': 234000000, 'webSearchUrl': 'https://www.bing.com/search?q=google.com'}, 'queryContext': {'originalQuery': 'google.com'}, '_type': 'SearchResponse'},
                {"_type": "SearchResponse", "queryContext": {"originalQuery": "qih6fywpvvie11loc5weawo2d.net"}, "rankingResponse": {}}
                ]
        return random.choice(data)

    def persist_data(self, data: dict):
        """
            Persist fetched data
        :param
            data: dict, data to be saved
        """
        with open("/opt/domain_analyzer/data/{}.json".format(self.domain), "w") as out_file:
            json.dump(data, out_file)

    def squared_sum(self, x):
        return round(sqrt(sum([a * a for a in x])), 3)

    def cos_similarity(self, x, y):
        numerator = sum(a * b for a, b in zip(x, y))
        denominator = self.squared_sum(x) * self.squared_sum(y)
        return round(numerator / float(denominator), 3)

    def tokenize(self, domain: str) -> list:
        return [self.charmap[c] for c in domain.lower().split(":")[0] if c in self.charmap]

    def similarity_preprocess(self, domains: list):
        source_vector = self.tokenize(self.domain)
        if domains:
            # x_data = []
            # for domain in domains:
            #     try:
            #         x_data.append(self.cos_similarity(source_vector, self.tokenize(domain)))
            #     except Exception as e:
            #         self.logger.warning("Failed to process data for {}: {}.".format(self.domain, e))
            #         raise Exception
            #     if len(x_data) >= self.max_domains_similarity:
            #         break
            checked_domains = [d for i, d in enumerate(domains) if d and i < self.max_domains_similarity]
            try:
                x_data = [self.cos_similarity(source_vector, self.tokenize(d)) for d in checked_domains]
            except Exception as e:
                self.logger.warning("Failed to process data for {}: {}.".format(self.domain, e))
                raise Exception
        else:
            x_data = []
        while len(x_data) != 20:
            x_data.append(2)
        return array([x_data])

    def blackbox_preprocess(self, domains: list):
        x_data = [self.tokenize(domain) for domain in domains]
        if len(x_data) < self.max_domains:
            while len(x_data) != self.max_domains:
                x_data.append([])
        else:
            x_data = x_data[:self.max_domains]
        return array(
            [pad_sequences(x_data, maxlen=self.max_domain_length, padding='post', truncating='post').astype("int8")])

    def embedding_preprocess(self, texts: list):
        x_data = self.tokenizer.texts_to_sequences(texts)
        if len(x_data) < self.max_texts:
            while len(x_data) != self.max_texts:
                x_data.append([])
        else:
            x_data = x_data[:self.max_texts]
        return array([pad_sequences(x_data, maxlen=self.max_text_length, padding='post', truncating='post')])

    def cleanup_text(self, text: str) -> str:
        text = text.replace("\n", "").encode('unicode_escape').decode('unicode_escape').strip().lower()
        if text.endswith("..."):
            text = text.replace("...", "")
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        for x in text:
            if x.isspace():
                continue
            if x in self.punctuations:
                text = text.replace(x, "")
            elif not x.isalnum():
                text = text.replace(x, "")
        text = re.sub(r'\s+', ' ', text)
        return remove_stopwords(text)

    def preprocess_data(self, data: dict) -> Tuple[int, list, list]:
        if ("results" in data and data["results"]) and (
                "results" in data["results"][0]["content"] and data["results"][0]["content"]["results"]):
            if "organic" in data["results"][0]["content"]["results"] and data["results"][0]["content"]["results"]["organic"]:
                domains, texts = [], []
                for org_res in data["results"][0]["content"]["results"]["organic"]:
                    domains.append(urlparse(org_res["url"]).netloc.replace("www.", ""))
                    texts.append(self.cleanup_text(org_res["desc"]))
                return len(data["results"][0]["content"]["results"]["organic"]), domains, texts
        return 0, [], []

    def classify_similarity(self, domains: list) -> int:
        return self.similarity_model.predict(self.similarity_preprocess(domains))[0]

    def classify_embedding(self, texts: list) -> int:
        prediction = self.cnn_texts.predict(self.embedding_preprocess(texts))
        self.nn_prob_logger.info("{},{},embedding".format(self.domain, prediction[0]))
        return 0 if prediction[0] <= self.nn_prob_threshold else 1

    def classify_blackbox(self, domains: list) -> int:
        prediction = self.cnn_blackbox.predict(self.blackbox_preprocess(domains))
        self.nn_prob_logger.info("{},{},blackbox".format(self.domain, prediction[0]))
        return 0 if prediction[0] <= self.nn_prob_threshold else 1

    def prepare_data(self) -> list:
        """
            Prepare data fro prediction
        :return:
            list, features used for prediction
        """
        raw_data = self.fetch_data()
        if not raw_data:
            raise NoDataException
        results, domains, texts = self.preprocess_data(raw_data)
        processed_metadata = [results, self.classify_similarity(domains), self.classify_embedding(texts),
                              self.classify_blackbox(domains)]
        self.result_logger.info("{},{}".format(self.domain, processed_metadata))
        return [processed_metadata]
