# from __future__ import absolute_import

import json
# import multiprocessing
import os
from datetime import datetime

import redis
from analyzer.evaluator import Evaluator
from analyzer.exc import FetchException, PreprocessException
from analyzer.preprocessor import Preprocessor
from analyzer.tools import build_logger
from celery import Celery, states
from celery import Task


# multiprocessing.set_start_method('spawn', force=True)


class DomainAnalyzer(Task):
    ignore_result = True
    name = "main.DomainAnalyzer"
    # max_retries = 6
    # retry_backoff = 300

    def __init__(self):
        self.logger = build_logger("main_worker", "/opt/domain_analyzer/logs/")
        self.connection = redis.Redis(os.environ["REDIS_RESULTS"], port=6379, db=os.environ["REDIS_DB"])
        self.result_connection = redis.Redis(os.environ["REDIS_ANALYSIS"], port=6379, db=os.environ["REDIS_DB_ANALYSIS"])
        if os.environ["MODE"] == "domain_analyzer":
            self.load_models()

    def load_models(self):
        from keras.models import load_model
        import gensim, pickle
        base_path = "/opt/domain_analyzer/analyzer/models/"
        # self.we_model._make_predict_function()
        # self.graph = tf.get_default_graph()
        self.tf_idf = pickle.load(open("{}tf_idf.pkl".format(base_path), "rb"))
        self.ensamble_tf_idf = pickle.load(open("{}ensamble_tf_idf.pkl".format(base_path), "rb"))
        self.lda_dictionary = gensim.corpora.Dictionary.load("{}lda_dictionary.pkl".format(base_path))
        self.lda_model = gensim.models.LdaMulticore.load("{}lda_model.pkl".format(base_path))
        self.ensamble_lda = pickle.load(open("{}ensamble_lda.pkl".format(base_path), "rb"))
        self.tokenizer = pickle.load(open("{}tokenizer.pkl".format(base_path), "rb"))
        self.ensamble_we = pickle.load(open("{}ensamble_we.pkl".format(base_path), "rb"))
        self.we_model = load_model("{}we_model.h5".format(base_path))
        self.scaler = pickle.load(open("{}scaler.pkl".format(base_path), "rb"))
        self.dense_model = load_model("{}dense_model.h5".format(base_path))
        # self.knn = pickle.load(open("{}knn.pkl".format(base_path), "rb"))
        # self.linearsvc = pickle.load(open("{}linearsvc.pkl".format(base_path), "rb"))
        # self.rforest = pickle.load(open("{}rforest.pkl".format(base_path), "rb"))
        # self.lightgbm = pickle.load(open("{}lightgbm.pkl".format(base_path), "rb"))

    def create_date(self):
        try:
            return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return "unknown"

    def persist_result(self, domain, result: list):
        try:
            self.connection.set(domain,
                                json.dumps(
                                    {"prediction": result[0], "probability": result[1], "created": self.create_date()}),
                                int(os.environ["RECORD_TTL"]))
        except Exception as e:
            self.logger.warning("Failed to persist results to Redis, {}".format(e))

    def analysis_done(self, domain: str):
        try:
            self.result_connection.delete(domain)
        except Exception as e:
            self.logger.warning("Failed to free domain from processing queue, {}".format(e))

    def run(self, domain):
        enrichers = []
        try:
            preprocessor = Preprocessor(domain, self.tf_idf, self.ensamble_tf_idf, self.lda_dictionary, self.lda_model,
                                        self.ensamble_lda, self.tokenizer, self.ensamble_we, self.we_model)
            domain_data = preprocessor.prepare_data()
            if len(enrichers) > 0:
                for enricher in enrichers:
                    domain_data[0].append(enricher.enrich(domain))
            evaluator = Evaluator(domain_data, domain, self.scaler, self.dense_model
                                  # self.knn, self.linearsvc, self.rforest, self.lightgbm
            )
            result = evaluator.predict_label()
            if result:
                self.persist_result(domain, result)
                self.analysis_done(domain)
                return result
            else:
                self.update_state(
                    state=states.FAILURE,
                    meta="Failed to get evaluation}"
                )
        except FetchException:
            if self.request.retries < 6:
                self.retry(args=(domain))
        except PreprocessException as pe:
            self.logger.warning(pe)
            self.update_state(
                state=states.FAILURE,
                meta="Preprocessor failed {}".format(pe)
            )
        except Exception as e:
            self.logger.warning("General failure during analysis, {}".format(e))
            self.update_state(
                state=states.FAILURE,
                meta=e
            )


app = Celery('oraculum', broker=os.environ.get("CELERY_BROKER"))
app.register_task(DomainAnalyzer())

#
# @app.task(bind=True, max_retries=6, retry_backoff=300)
# def predict_domain(self, domain: id):
#     """
#     Main execution method.
#     """
#     logger = logging.getLogger("main")


if __name__ == '__main__':
    # env_variable_validation()
    app.start()
