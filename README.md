Domain analyzer
=========
The purpose of this repository is to provide a container for a domain analysis. The analysis evaluate whether the domain is 
mailicious or not and uses a search engine data for it. For running the code you would need Docker. To start use the convenience **docker-compose.yml** in root of this project.

Requirements
=========
The only thing a user needs for setup is **Bing** api key, which would get the data. If key is not provided a **dry-run** option can be used.
In that scenario a random choice of two domain data is presented to the user. The option to not use Docker is not unspported at the moment however the user 
may try to get it running using the information in Dockerfile and docker-compose.


Behaviour
=========
The user interacts with the system via REST API. Example of the request mich be something like this:

curl -X POST http://localhost:8000/predict -d '{"domain": "google.com"}'

The communication with the api is described in [Swagger](https://app.swaggerhub.com/apis/Narzhan/Oraculum/1.0.0)  

Envs
----------

#### Domain_analyzer 
- CELERY_BROKER: address of Redis broker ('redis://localhost:6379/3')
- MODE: service name for startup script
- REDIS_RESULTS: address of a redis result cache
- REDIS_ANALYSIS: address of a redis analysis status cache
- REDIS_DB: redis result db
- REDIS_DB_ANALYSIS: redis analysis status db
- RECORD_TTL: ttl of the analysis record
- BING_API_KEY: api key for bing [Conginitve api](https://azure.microsoft.com/en-us/services/cognitive-services/)
- PERSIST_DATA: the option to persist the data for future use (retraining) default path is **/opt/domain_analyzer/data/**, if not desired the env should not be supplied
- KERAS_BACKEND: specify backend for Keras, currently Theano because of  concurrency issue
- TEST_MODE: the option whether to not download data from Bing, if not desired the env should not be supplied
 
#### Api 
- CELERY_BROKER: address of Redis broker ('redis://localhost:6379/3/')
- MODE: service name for startup script
- REDIS_RESULTS: address of a redis result cache
- REDIS_ANALYSIS: address of a redis analysis status cache
- REDIS_DB: redis result db
- REDIS_DB_ANALYSIS: redis analysis status db


Redis dbs:
----------
- 2: Analysis results
- 3: Celery broker data
- 4: Analysis status