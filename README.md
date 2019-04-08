Domain analyzer
=========
The purpose of this repository is to provide a system for a domain analysis. The system evaluates whether the domain is 
malicious or not and uses a search engine data for it. For running the code you would need Docker. To start use the convenience **docker-compose.yml** in root of this project.

Requirements
=========
The only thing a user needs for setup is **Bing** api key, which would get the data. If key is not provided a **dry-run** option can be used.
In that scenario a random choice of data for two domain data is presented to the user. The option to not use Docker is not supported at the moment however the user 
may try to get it running using the information in Dockerfile and docker-compose.

Test Mode
=========
The system supports a dry run option called **Test mode**. It allows a test usage without the need for Bing api key. If used, upon receiving a request one
of the two pre loaded data samples is selected. These samples are data from Bing of two domains one is malicious and one is clean from a one point of time.   
When a request for analysis is invoked a random one of the two samples is selected. Therefore it is possible to get malicious analysis result for clean domain and vice versa.


Usage
=========
To try it out just run following command in the root of the repo. The preconfigured docker-compose is set up with dry-run so no key or set up is necessary.

`docker-compose up -d`

Behaviour
=========
The user interacts with the system via REST API using HTTP POST requests. Example of the request mich be something like this:

`curl -X POST http://localhost:8000/predict -d '{"domain": "google.com"}'`

The communication with the api is described in [Swagger](https://app.swaggerhub.com/apis/Narzhan/Oraculum/1.0.0)  

Environment variables
----------
The system allows configuration using following environment variables


#### Domain_analyzer 
- CELERY_BROKER: address of Redis broker ('redis://localhost:6379/3')
- MODE: service name for startup script
- CACHE_RESULTS: address of a result cache
- CACHE_ANALYSIS: address of a analysis status cache
- RESULTS_DB: result db while using Redis
- ANALYSIS_DB: analysis status db  while using Redis
- RECORD_TTL: ttl of the analysis record
- BING_API_KEY: api key for bing [Conginitve api](https://azure.microsoft.com/en-us/services/cognitive-services/)
- PERSIST_DATA: the option to persist the data for future use (retraining) default path is **/opt/domain_analyzer/data/**, if needed the value should be **true** if not it should be **false** or not supplied
- KERAS_BACKEND: specify backend for **Keras**, currently **Theano** because of concurrency issue
- TEST_MODE: the option whether to not download data from Bing, if needed the value should be **true** if not it should be **false** or not supplied
 
#### Api 
- CELERY_BROKER: address of Redis broker ('redis://localhost:6379/3/')
- MODE: service name for startup script
- CACHE_RESULTS: address of a redis result cache
- CACHE_ANALYSIS: address of a redis analysis status cache
- RESULTS_DB: result db while using Redis
- ANALYSIS_DB: analysis status db  while using Redis


Cache (Redis) dbs:
----------
- 2: Analysis results
- 3: Celery broker data
- 4: Analysis status
