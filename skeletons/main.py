import os
from celery import Celery
from celery import Task


class DomainAnalyzer(Task):
    ignore_result = True
    name = "main.DomainAnalyzer"

    # max_retries = 6
    # retry_backoff = 300

    def run(self, domain):
        pass


app = Celery('oraculum', broker=os.environ.get("CELERY_BROKER"))
app.register_task(DomainAnalyzer())

if __name__ == '__main__':
    app.start()
