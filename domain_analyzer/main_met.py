from __future__ import absolute_import
import os
import logging
import pickle
import redis
from datetime import datetime
import json
from celery import Celery
from analyzer.preprocessor import Preprocessor
from analyzer.evaluator import Evaluator
from analyzer.tools import build_logger
from analyzer.exc import FetchException, PreprocessException
from keras.models import load_model
import gensim

# def env_variable_validation():
#     variables = ["MAIL_SMTP", "MAIL", "MAIL_PASSWORD", "SENDER_NUMBER", "CELERY_BROKER", "ERROR_API", "SMS_API",
#                  "JWT_SECRET", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_DATABASE", "RETAIL_API"]
#     for variable in variables:
#         if variable not in os.environ:
#             print("Environment variable {} is missing but is required".format(variable))
#             raise KeyboardInterrupt
if os.environ["MODE"] == "domain_analyzer":
    base_path = "/opt/domain_analyzer/analyzer/models/"
    we_model = load_model("{}we_model.h5".format(base_path))
    tf_idf = pickle.load(open("{}tf_idf.pkl".format(base_path), "rb"))
    ensamble_tf_idf = pickle.load(open("{}ensamble_tf_idf.pkl".format(base_path), "rb"))
    lda_dictionary = gensim.corpora.Dictionary.load("{}lda_dictionary.pkl".format(base_path))
    lda_model = gensim.models.LdaMulticore.load("{}lda_model.pkl".format(base_path))
    ensamble_lda = pickle.load(open("{}ensamble_lda.pkl".format(base_path), "rb"))
    tokenizer = pickle.load(open("{}tokenizer.pkl".format(base_path), "rb"))
    ensamble_we = pickle.load(open("{}ensamble_we.pkl".format(base_path), "rb"))

app = Celery('oraculum', broker=os.environ.get("CELERY_BROKER"))

# def create_date():
#     try:
#         return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
#     except Exception:
#         return "unknown"
#
#
# def persist_result(domain, connection, result: list):
#     logger = logging.getLogger("main")
#     try:
#         connection.set(domain,
#                        json.dumps({"prediction": result[0], "probability": result[1], "created": create_date()}),
#                        int(os.environ["RECORD_TTL"]))
#     except Exception as e:
#         logger.warning("Failed to persist results to Redis, {}".format(e))
#
#
# def analysis_done(connection, domain: str):
#     logger = logging.getLogger("main")
#     try:
#         connection.delete(domain)
#     except Exception as e:
#         logger.warning("Failed to free domain from processing queue, {}".format(e))


@app.task(bind=True, max_retries=6, retry_backoff=300)
def predict_domain(self, domain: id):
    """
    Main execution method.
    """
    logger = logging.getLogger("main")
    enrichers = []
    # connection = redis.Redis(os.environ["REDIS_RESULTS"], port=6379, db=os.environ["REDIS_DB"])
    # result_connection = redis.Redis(os.environ["REDIS_ANALYSIS"], port=6379, db=os.environ["REDIS_DB_ANALYSIS"])
    try:
        logger.info("Data received")
        domain_data = Preprocessor(domain, tf_idf, ensamble_tf_idf, lda_dictionary, lda_model,
                                   ensamble_lda, tokenizer, we_model, ensamble_we).prepare_data()
        logger.info("Domain preprocessed, {}".format(domain_data))
        if len(enrichers) > 0:
            for enricher in enrichers:
                domain_data.append(enricher.enrich(domain))
        result = Evaluator(domain_data).predict_label()
        logger.info("Result present, {}".format(result))
        # persist_result(domain, connection, result)
        # analysis_done(result_connection, domain)
        return result
    except FetchException:
        if self.request.retries < 6:
            self.retry(args=(domain))
    except PreprocessException as pe:
        logger.warning(pe)
    except Exception as e:
        logger.warning("General failure during analysis, {}".format(e))


if __name__ == '__main__':
    logger = build_logger("main", "/opt/domain_analyzer/logs/")
    # env_variable_validation()
    app.start()
