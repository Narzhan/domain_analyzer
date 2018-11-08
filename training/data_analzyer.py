import json
import os
from tld import get_tld
from datetime import datetime, timedelta


class Analyzer:
    def __init__(self):
        self.result = {}

    def read_data(self, path: str, label: int):
        for file in os.listdir(path):
            try:
                with open("{}{}".format(path, file), "r") as data:
                    domain = file.replace(".json", "")
                    self.result[domain] = {}
                    self.get_features(json.load(data), domain, label)
                    self.dump_result()
            except Exception as e:
                print("File error, {}, {}".format(file, e))
            finally:
                self.result = {}

    # def analyze_domain(self, domain: str, data: dict, label: int =0):
    #     self.result[domain] = {}
    #     self.get_features(data, domain, label)

    def load_data(self):
        self.dump_header()
        for path, label in {"": 0,
                            "": 1}.items():
            self.read_data(path, label)


    def get_features(self, source: dict, domain: str, label: int):
        sections = {"webPages": self.web_pages, "relatedSearches": self.related_searches,
                    "rankingResponse": self.ranking_response}
        for section, method in sections.items():
            try:
                method(source[section], domain, label)
            except KeyError:
                method({}, domain, label)

    def web_pages(self, source: dict, domain: str, label: int):
        categories = ["totalEstimatedMatches", "someResultsRemoved"]
        full_match = 0
        part_math = 0
        page_count = 0
        about = 0
        deep_links = 0
        fresh = 0
        infection = 0
        if source != {}:
            for cat in categories:
                try:
                    self.result[domain][cat] = source[cat]
                except KeyError:
                    self.result[domain][cat] = 0
            try:
                self.result[domain]["pages"] = len(source["value"])
            except KeyError:
                self.result[domain]["pages"] = 0
            try:
                domain_tld = get_tld(domain, as_object=True, fix_protocol=True)
                for page in source["value"]:
                    try:
                        url_tld = get_tld(page["url"], as_object=True, fix_protocol=True)
                    except Exception:
                        pass
                    else:
                        if domain_tld.fld == url_tld.fld:
                            part_math += 1
                            page_count += 1
                            if domain_tld.subdomain == url_tld.subdomain:
                                full_match += 1
                            if "about" in page:
                                about += 1
                            if "deepLinks" in page:
                                deep_links += 1
                            if "dateLastCrawled" in page and \
                                    datetime.now() - datetime.strptime(page["dateLastCrawled"].split("T")[0],
                                                                       "%Y-%m-%d") < timedelta(days=7):
                                fresh += 1
                            if "snippet" in page:
                                infection += self.text_analyzer(page["snippet"])
                        else:
                            for word in ["virustotal", "sandbox", "malwr", "hybrid-analysis"]:
                                if word in page["url"]:
                                    infection += 1
            except Exception as e:
                print(e)
            self.result[domain].update(
                {"full_path": full_match, "part_path": part_math, "page_count": page_count, "about": about,
                 "deep_links": deep_links, "fresh": fresh, "infection": infection, "label": label})
        else:
            self.result[domain].update(
                {"full_path": full_match, "part_path": part_math, "page_count": page_count, "about": about,
                 "deep_links": deep_links, "fresh": fresh, "infection": infection, "pages": 0,
                 "totalEstimatedMatches": 0, "someResultsRemoved": 0, "label": label})

    def related_searches(self, source: dict, domain: str, label: int):
        if source != {}:
            self.result[domain]["related_searches"] = len(source["value"])
        else:
            self.result[domain]["related_searches"] = 0

    def ranking_response(self, source: dict, domain: str, label: int):
        if source != {}:
            self.result[domain]["ranking_response"] = len(source["mainline"]["items"])
        else:
            self.result[domain]["ranking_response"] = 0

    def text_analyzer(self, source: set) -> int:
        for word in ['phishing', 'ddos', "c&c", "sample", 'spam', 'scanner', 'dropzone', 'malware',
                     'botnet drone', 'ransomware',
                     'dga', 'exploit', 'brute-force', 'ids alert', 'defacement',
                     'compromised', 'backdoor', 'vulnerable service', 'blacklist', "bot", "trojan", "spam", "virus",
                     "backdoor"]:
            if word in source:
                return 1
        return 0

    def dump_header(self):
        header = ["ranking_response", "related_searches", 'full_path', 'part_path', 'page_count', 'about',
                  'deep_links', 'fresh', 'infection', 'pages', 'totalEstimatedMatches', 'someResultsRemoved', 'label',
                  "domain"]
        with open("test_data.csv", "w") as file:
            file.write("{}\n".format(",".join(header)))

    def dump_result(self):
        header = ["ranking_response", "related_searches", 'full_path', 'part_path', 'page_count', 'about',
                  'deep_links', 'fresh', 'infection', 'pages', 'totalEstimatedMatches', 'someResultsRemoved', 'label']
        with open("test_data.csv", "a") as file:
            for domain, data in self.result.items():
                for place in header:
                    file.write("{},".format(data[place]))
                file.write(domain)
                file.write("\n")


if __name__ == '__main__':
    analyzer = Analyzer()
    analyzer.load_data()
