import os
import traceback
from analyzer.evaluator import Evaluator
from analyzer.exc import FetchException, PreprocessException, NoDataException, QuotaReached
from analyzer.preprocessor import Preprocessor
from analyzer.tools import build_logger
from analyzer.cache import CacheConnector
from celery import Celery, states
from celery import Task


# def env_variable_validation():
#     variables = ["MAIL_SMTP", "MAIL", "MAIL_PASSWORD", "SENDER_NUMBER", "CELERY_BROKER", "ERROR_API", "SMS_API",
#                  "JWT_SECRET", "DB_USER", "DB_PASSWORD", "DB_HOST", "DB_PORT", "DB_DATABASE", "RETAIL_API"]
#     for variable in variables:
#         if variable not in os.environ:
#             print("Environment variable {} is missing but is required".format(variable))
#             raise KeyboardInterrupt
if os.environ["MODE"] == "domain_analyzer":
    # import tensorflow as tf
    import pickle

    base_path = "/opt/domain_analyzer/analyzer/models/"
    ensemble = pickle.load(open("{}ensemble_model_v2.pkl".format(base_path), "rb"))
    ensemble_scaler = pickle.load(open("{}ensemble_scaler_v2.pkl".format(base_path), "rb"))
    similarity_model = pickle.load(open("{}gb_similarity.pkl".format(base_path), "rb"))
    tokenizer = pickle.load(open("{}tokenizer.pkl".format(base_path), "rb"))
    # cnn_blackbox = tf.keras.models.load_model("{}domains_blackbox_no_embedding_v2.h5".format(base_path))
    # cnn_texts = tf.keras.models.load_model("{}texts_we_glove_cnn_v3.h5".format(base_path))

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
    logger = build_logger("main_worker", "/opt/domain_analyzer/logs/")
    cache = CacheConnector()
    try:
        preprocessor = Preprocessor(domain, similarity_model, tokenizer)
        domain_data = preprocessor.prepare_data()
        evaluator = Evaluator(domain_data, domain, ensemble_scaler, ensemble)
        result = evaluator.predict_label()
        if isinstance(result, int):
            cache.push_result(domain, result)
            cache.finish_analysis(domain)
            return result
        else:
            raise Exception
    except QuotaReached:
        cache.finish_analysis(domain)
        cache.push_result(domain, -1)
        return "Quota reached, try later."
    except FetchException:
        if self.request.retries < 6:
            self.retry(args=(domain))
    except PreprocessException as pe:
        cache.finish_analysis(domain)
        logger.warning(pe)
        self.update_state(
            state=states.FAILURE,
            meta="Preprocessor failed {}".format(pe)
        )
    except NoDataException:
        cache.finish_analysis(domain)
        return "No data found"
    except Exception as e:
        cache.finish_analysis(domain)
        logger.warning("General failure during analysis, {}".format(e))
        logger.warning("{}".format(traceback.format_exc()))
        self.update_state(
            state=states.FAILURE,
            meta=e
        )


if __name__ == '__main__':
    app.start()
