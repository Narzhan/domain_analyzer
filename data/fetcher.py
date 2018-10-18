import requests
import os


class Fetcher:
    def __init__(self):
        try:
            self.search_url = os.environ["API_URL"]
        except KeyError:
            self.search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"
        try:
            self.headers = {"Ocp-Apim-Subscription-Key": os.environ["API_KEY"]}
        except KeyError:
            self.headers = {"Ocp-Apim-Subscription-Key": "0192313657c949839f025d9d6f027d17"}

    def pull_data(self, domain: str) -> dict:
        params = {"q": "{}+site:{}".format(domain, domain), "textDecorations": True, "textFormat": "HTML"}
        try:
            request = requests.get(self.search_url, headers=self.headers, params=params)
        except requests.exceptions.RequestException as e:
            print()
            return {}
        else:
            return request.json()


