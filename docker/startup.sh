#!/usr/bin/env bash
if [ $MODE = "domain_analyzer" ]
then
    celery -A main worker --autoscale=6,1 --loglevel=info
else
    gunicorn -b 0.0.0.0:8000 --access-logfile /opt/domain_analyzer/logs/api_access.log app:app
fi