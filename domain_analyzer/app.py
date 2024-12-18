import json
import re

import falcon
# from main import app as celery_app
# from main import predict_domain
from analyzer.tools import build_logger
from analyzer.cache import CacheConnector
# from main import DomainAnalyzer
from scheduler import predict_domain

# ^(?=.*[^\.]$)((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.?){4}$
class Predict:
    def __init__(self):
        self.logger = build_logger("api", "/opt/domain_analyzer/logs/")
        self.domain_pattern = re.compile("(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z0-9][a-z0-9-]{0,61}[a-z]")
        self.cache = CacheConnector()

    # def validate_input(self, domain: str) -> bool:
    #     return True if self.domain_pattern.match(domain) else False

    def create_task(self, domain: str):
        # DomainAnalyzer().delay(domain)
        predict_domain.delay(domain)
        self.cache.create_analysis(domain)
        return {"status": "Analysis submitted"}, falcon.HTTP_201

    def handle_request(self, request: dict):
        if "force" in request and request["force"] is True:
            return self.create_task(request["domain"])
        if self.cache.check_result(request["domain"]):
            result = self.cache.fetch_result(request["domain"])
            if result["prediction"] == -1:
                return {"status": "Quota reached."}, falcon.HTTP_429
            else:
                return result, falcon.HTTP_200
        else:
            if not self.cache.check_analysis(request["domain"]):
                return self.create_task(request["domain"])
            else:
                return {"status": "Analysis pending"}, falcon.HTTP_201

    def on_post(self, req, resp):
        try:
            analysis = json.loads(req.stream.read().decode("utf-8"))
        except json.JSONDecodeError as de:
            resp.status = falcon.HTTP_400
            resp.media = {"error": "incorrect json format, {}".format(de)}
        else:
            if self.domain_pattern.match(analysis["domain"]):
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


app = falcon.App()
app.add_route("/predict", Predict())
app.add_route("/status", Healthcheck())
