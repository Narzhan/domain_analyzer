#!/usr/bin/env bash
if [ $MODE = "domain_analyzer" ]
then
    celery -A main worker --autoscale=6,1 --loglevel=info
else
    gunicorn -b 0.0.0.0:8000 --worker-class gevent --access-logfile "-" app:app
fi