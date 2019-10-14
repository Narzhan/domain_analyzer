import os
import traceback
from analyzer.evaluator import Evaluator
from analyzer.exc import FetchException, PreprocessException
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
        from keras.models import load_model
        import gensim, pickle
        base_path = "/opt/domain_analyzer/analyzer/models/"
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
        self.knn = pickle.load(open("{}knn.pkl".format(base_path), "rb"))
        self.linearsvc = pickle.load(open("{}linearsvc.pkl".format(base_path), "rb"))
        self.rforest = pickle.load(open("{}rforest.pkl".format(base_path), "rb"))

    def run(self, domain):
        """
            Main celery worker which analysses the domain
        :param
         domain: str, domain to be queried
        """
        enrichers = []
        try:
            preprocessor = Preprocessor(domain, self.tf_idf, self.ensamble_tf_idf, self.lda_dictionary, self.lda_model,
                                        self.ensamble_lda, self.tokenizer, self.ensamble_we, self.we_model)
            domain_data = preprocessor.prepare_data()
            if len(enrichers) > 0:
                for enricher in enrichers:
                    domain_data[0].append(enricher.enrich(domain))
            evaluator = Evaluator(domain_data, domain, self.scaler, self.dense_model, self.knn, self.linearsvc,
                                  self.rforest)
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
        except Exception as e:
            self.cache.finish_analysis(domain)
            self.logger.warning("General failure during analysis, {}".format(e))
            logger.warning("{}".format(traceback.format_exc()))
            self.update_state(
                state=states.FAILURE,
                meta=e
            )


app = Celery('oraculum', broker=os.environ.get("CELERY_BROKER"))
app.register_task(DomainAnalyzer())

if __name__ == '__main__':
    app.start()
