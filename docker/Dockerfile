FROM ubuntu:18.04
MAINTAINER Václav Brůžek <vaclav.bruzek@whalebone.io>

RUN apt-get update -y && \
    apt-get install -y python3-pip nano

RUN pip3 --no-cache-dir install requests gunicorn celery falcon redis scikit-learn==0.20.0 keras theano pandas gensim tld h5py nltk lightgbm

RUN mkdir -p /opt/domain_analyzer/logs/
WORKDIR /opt/domain_analyzer/
COPY analyzer/models/* analyzer/models/
COPY app.py startup.sh main.py ./
COPY analyzer/*.py analyzer/

CMD ["bash", "startup.sh"]
