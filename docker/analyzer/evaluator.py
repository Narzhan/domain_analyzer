import os
import pickle

from .tools import build_logger


class Evaluator:
    def __init__(self, data: list, domain: str, scaler, dense_model, knn, linersvc, rforrest, ):
        self.domain = domain
        self.model_path = "/opt/domain_analyzer/analyzer/models/"
        self.logger = build_logger("evaluator", "/opt/domain_analyzer/logs/")
        self.scaler = scaler
        self.dense_model = dense_model
        self.knn = knn
        self.linearsvc = linersvc
        self.rforrest = rforrest
        self.data = self.scale_data(data)
        self.prepare_results()

    def scale_data(self, data: list) -> list:
        # scaler = pickle.load(open("{}scaler.pkl".format(self.model_path), "rb"))
        return self.scaler.transform(data)

    def prepare_results(self):
        try:
            if not os.path.exists("/opt/domain_analyzer/logs/results.csv"):
                with open("/opt/domain_analyzer/logs/results.csv", "w") as file:
                    file.write("domain,knn,knn_prob,lsvc,rforest,rforest_prob,lightgbm,lightgbm_prob,nn,nn_prob\n")
        except Exception as e:
            self.logger.info("Failed to create test file, {}".format(e))

    def persist_results(self, results: list):
        try:
            with open("/opt/domain_analyzer/logs/results.csv", "a") as file:
                file.write("{},{}\n".format(self.domain, ",".join(str(result) for result in results)))
        except Exception as e:
            self.logger.info("Failed to append to test file, {}".format(e))

    def predict_domain(self, model_name: str, classifier) -> list:
        try:
            if classifier is None:
                classifier = pickle.load(open("{}{}.pkl".format(self.model_path, model_name), "rb"))
            prediction = int(classifier.predict(self.data)[0])
        except Exception as e:
            self.logger.warning("Failed to predict with algortihm {}, {}".format(model_name, e))
        else:
            if model_name != "linearsvc":
                return [prediction, max(classifier.predict_proba(self.data)[0])]
            return [prediction]

    def predict_nn(self) -> list:
        try:
            # nn = load_model("{}dense_model.h5".format(self.model_path))
            prediction = self.dense_model.predict(self.data)[0][0]
        except Exception as e:
            self.logger.warning("Failed to predict with nn, {}".format(e))
        else:
            return [int(prediction > 0.5), float(prediction)]

    # {"knn": self.knn, "linearsvc": self.linearsvc, "lightgbm": self.lightgbm,
    #  "rforest": self.rforrest}.items()

    def predict_label(self):
        results = []
        for model_name, model in {"knn": self.knn, "linearsvc": self.linearsvc, "lightgbm": None,
                           "rforest": self.rforrest}.items():
            if model_name != "lightgbm":
                results.extend(self.predict_domain(model_name, model))
            else:
                prediction = self.predict_domain(model_name, model)
                results.extend(prediction)
        results.extend(self.predict_nn())
        self.logger.info("{} - {}".format(self.domain, results))
        self.persist_results(results)
        if "prediction" in locals():
            return prediction
        else:
            return None