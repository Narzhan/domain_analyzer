class Enricher:

    def __init__(self, domain: str):
        self.domain = domain

    def enrich(self) -> list:
        """
            Enrich domain data from a predefined source
        :return:
            list with a single feature
        """
        return []