import os
import traceback
from analyzer.evaluator import Evaluator
from analyzer.exc import FetchException, PreprocessException, NoDataException
from analyzer.preprocessor import Preprocessor
from analyzer.tools import build_logger
from analyzer.cache import CacheConnector
from celery import Celery, states
from celery import Task


class DomainAnalyzer(Task):
    ignore_result = True
    name = "main.DomainAnalyzer"

    # max_retries = 6
    # retry_backoff = 300

    def __init__(self):
        self.logger = build_logger("main_worker", "/opt/domain_analyzer/logs/")
        self.cache = CacheConnector()
        if os.environ["MODE"] == "domain_analyzer":
            self.load_models()

    def load_models(self):
        """
            Load pretrained models which should be passed as reference
        :return:
        """
        import tensorflow as tf
        import pickle
        base_path = "/opt/domain_analyzer/analyzer/models/"
        self.ensemble = pickle.load(open("{}ensemble_model_v2.pkl".format(base_path), "rb"))
        self.ensemble_scaler = pickle.load(open("{}ensemble_scaler_v2.pkl".format(base_path), "rb"))
        self.similarity_model = pickle.load(open("{}gb_similarity.pkl".format(base_path), "rb"))
        self.tokenizer = pickle.load(open("{}tokenizer.pkl".format(base_path), "rb"))
        self.cnn_blackbox = tf.keras.models.load_model("{}domains_blackbox_no_embedding_v2.h5".format(base_path))
        self.cnn_texts = tf.keras.models.load_model("{}texts_we_glove_cnn_v3.h5".format(base_path))

    def run(self, domain):
        """
            Main celery worker which analysses the domain
        :param
         domain: str, domain to be queried
        """
        # enrichers = []
        try:
            preprocessor = Preprocessor(domain, self.similarity_model, self.cnn_blackbox, self.cnn_texts, self.tokenizer)
            domain_data = preprocessor.prepare_data()
            evaluator = Evaluator(domain_data, domain, self.ensemble_scaler, self.ensemble)
            result = evaluator.predict_label()
            if result:
                self.cache.push_result(domain, result)
                self.cache.finish_analysis(domain)
                return result
            else:
                raise Exception
        except FetchException:
            if self.request.retries < 6:
                self.retry(args=(domain))
        except PreprocessException as pe:
            self.cache.finish_analysis(domain)
            self.logger.warning(pe)
            self.update_state(
                state=states.FAILURE,
                meta="Preprocessor failed {}".format(pe)
            )
        except NoDataException:
            self.cache.finish_analysis(domain)
            return "No data found"
        except Exception as e:
            self.cache.finish_analysis(domain)
            self.logger.warning("General failure during analysis, {}".format(e))
            self.logger.warning("{}".format(traceback.format_exc()))
            self.update_state(
                state=states.FAILURE,
                meta=e
            )


app = Celery('oraculum', broker=os.environ.get("CELERY_BROKER"))
app.register_task(DomainAnalyzer())

if __name__ == '__main__':
    app.start()
