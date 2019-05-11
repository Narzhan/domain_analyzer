import json
import os
from datetime import datetime
import redis

from .tools import build_logger


class CacheConnector:

    def __init__(self):
        self.result_connection = redis.Redis(os.environ["CACHE_RESULTS"], port=6379,
                                             db=os.environ["RESULTS_DB"])
        self.analysis_connection = redis.Redis(os.environ["CACHE_ANALYSIS"], port=6379,
                                               db=os.environ["ANALYSIS_DB"])
        self.logger = build_logger("cache", "/opt/domain_analyzer/logs/")

    def fetch_result(self, domain: str) -> dict:
        """
        Get domain analysis results from cache
        :param
            domain: str, queried domain
        :return:
            analysis: dict, analysis
        """
        try:
            domain_analysis = json.loads(self.result_connection.get(domain).decode("utf-8", errors="ignore"))
            domain_analysis.update({"domain": domain})
            return domain_analysis
        except Exception as e:
            self.logger.warning("Failed to get cached results, {}".format(e))
            return {"status": "Cache error with domain {}".format(domain)}

    def push_result(self, domain: str, result: list):
        """
            Save analysis result to cache
        :params
             domain: str, domain name
             result: list, prediction and prediction probability
        """
        try:
            self.result_connection.set(domain,
                                json.dumps(
                                    {"prediction": result[0], "probability": result[1], "created": self.create_date()}),
                                int(os.environ["RECORD_TTL"]))
        except Exception as e:
            self.logger.warning("Failed to persist results to cache for domain {}, {}".format(domain, e))

    def check_result(self, domain) -> bool:
        """
            Check if domain result is in cache
        :param
            domain: str, queried doamin
        :return:
            bool: True if present, else False
        """
        try:
            return True if self.result_connection.exists(domain) else False
        except Exception as e:
            self.logger.warning("Failed to get result status, {}".format(e))
            return False

    def create_analysis(self, domain: str):
        """
            Create analysis created record to be checked for running analysis
        :param
            domain: str, name
        """
        try:
            self.analysis_connection.set(domain, "running", 300)
        except Exception as e:
            self.logger.warning("Failed to create analysis status, {}".format(e))

    def check_analysis(self, domain) -> bool:
        """
            Check analysis progress for queried domain
        :param
            domain: str, name
        """
        try:
            return True if self.analysis_connection.exists(domain) else False
        except Exception as e:
            self.logger.warning("Failed to get analysis status, {}".format(e))
            return False

    def finish_analysis(self, domain: str):
        """
            Clear analysis status
        :param
            domain: str, domain name
        """
        try:
            self.analysis_connection.delete(domain)
        except Exception as e:
            self.logger.warning("Failed to free domain from processing queue, {}".format(e))

    def create_date(self) -> str:
        """
            Get current timestamp in ISO-8601 format for analysis result
        :return:
            str: current timestamp
        """
        try:
            return datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return "unknown"
