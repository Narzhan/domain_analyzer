from datetime import datetime


class CacheConnector:

    def fetch_result(self, domain: str) -> dict:
        """
        Get domain analysis results from cache
        :param
            domain: str, queried domain
        :return:
            analysis: dict, analysis
        """
        pass

    def push_result(self, domain: str, result: list):
        """
            Save analysis result to cache
        :params
             domain: str, domain name
             result: list, prediction and prediction probability
        """
        pass

    def check_result(self, domain) -> bool:
        """
            Check if domain result is in cache
        :param
            domain: str, queried doamin
        :return:
            bool: True if present, else False
        """
        pass

    def create_analysis(self, domain: str):
        """
            Create analysis created record to be checked for running analysis
        :param
            domain: str, name
        """
        pass

    def check_analysis(self, domain) -> bool:
        """
            Check analysis progress for queried domain
        :param
            domain: str, name
        """
        pass

    def finish_analysis(self, domain: str):
        """
            Clear analysis status
        :param
            domain: str, domain name
        """
        pass

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
