FROM ubuntu:20.04
MAINTAINER Václav Brůžek <vaclav.bruzek@whalebone.io>

RUN apt-get update -y && \
    apt-get install -y python3-pip nano

RUN pip3 --no-cache-dir install requests gunicorn celery falcon redis gensim "scikit-learn==1.0.2" "tensorflow==2.8.0" "pandas==1.4.1" gevent

RUN mkdir -p /opt/domain_analyzer/logs/ /opt/domain_analyzer/data/
WORKDIR /opt/domain_analyzer/
COPY analyzer/models/* analyzer/models/
COPY app.py startup.sh scheduler.py main.py ./
COPY analyzer/*.py analyzer/

CMD ["bash", "startup.sh"]
