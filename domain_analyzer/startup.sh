#!/usr/bin/env bash
if [ $MODE = "domain_analyzer" ]
then
    celery -A scheduler worker --autoscale=6,1 --loglevel=info
else
    gunicorn -b 0.0.0.0:8000 --worker-class gevent app:app
fi