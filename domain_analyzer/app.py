import json
import os
import re

import falcon
import redis
# from main import app as celery_app
# from main import predict_domain
from analyzer.tools import build_logger
from analyzer.cache import CacheConnector
from main import DomainAnalyzer


class Predict:
    def __init__(self):
        # self.result_connection = redis.Redis(os.environ["REDIS_RESULTS"], port=6379,
        #                                      db=os.environ["REDIS_DB"])
        # self.analysis_connection = redis.Redis(os.environ["REDIS_ANALYSIS"], port=6379,
        #                                        db=os.environ["REDIS_DB_ANALYSIS"])
        self.logger = build_logger("api", "/opt/domain_analyzer/logs/")
        self.domain_pattern = re.compile("(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z]")
        self.cache = CacheConnector()

    # def fetch_result(self, domain: str) -> dict:
    #     try:
    #         domain_analysis = json.loads(self.result_connection.get(domain).decode("utf-8", errors="ignore"))
    #         domain_analysis.update({"domain": domain})
    #         return domain_analysis
    #     except Exception as e:
    #         self.logger.warning("Failed to get cached results, {}".format(e))
    #         return {"status": "Cache error with domain {}".format(domain)}
    #
    # def check_cache(self, connection, domain) -> bool:
    #     try:
    #         return True if connection.exists(domain) else False
    #     except Exception as e:
    #         self.logger.warning("Failed to get analysis status, {}".format(e))
    #         return False

    def validate_input(self, domain: str) -> bool:
        return True if self.domain_pattern.match(domain) else False

    # def analysis_created(self, domain: str):
    #     try:
    #         self.analysis_connection.set(domain, "running", 300)
    #     except Exception as e:
    #         self.logger.warning("Failed to create analysis status, {}".format(e))

    def create_task(self, domain: str):
        DomainAnalyzer().delay(domain)
        self.cache.create_analysis(domain)
        return {"status": "Analysis submitted"}, falcon.HTTP_201

    def handle_request(self, request: dict):
        if "force" in request and request["force"] is True:
            return self.create_task(request["domain"])
        if self.cache.check_result(request["domain"]):
            return self.cache.fetch_result(request["domain"]), falcon.HTTP_200
        else:
            if not self.cache.check_analysis(request["domain"]):
                return self.create_task(request["domain"])
            else:
                return {"status": "Analysis pending"}, falcon.HTTP_201

    def on_post(self, req, resp):
        try:
            analysis = json.loads(req.stream.read(req.content_length or 0).decode("utf-8"))
        except json.JSONDecodeError as de:
            resp.status = falcon.HTTP_400
            resp.media = {"error": "incorrect json format, {}".format(de)}
        else:
            if self.validate_input(analysis["domain"]):
                try:
                    status, response_code = self.handle_request(analysis)
                except Exception as e:
                    resp.status = falcon.HTTP_500
                    resp.media = {"status": "Failed to create domain processing job", "reason": "{}".format(e)}
                else:
                    resp.status = response_code
                    resp.media = status
            else:
                resp.status = falcon.HTTP_400
                resp.media = {"error": "Only valid domain names are supported"}


class Healthcheck(object):
    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.media = {"status": "ok"}


app = falcon.API()
app.add_route("/predict", Predict())
app.add_route("/status", Healthcheck())
