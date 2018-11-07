
import itertools, requests, json, os
import multiprocessing.pool


def worker(domain):
    # print(domain)
    # return domain
    try:
        params = {"q": domain, "textDecorations": True, "textFormat": "HTML"}
        headers = {"Ocp-Apim-Subscription-Key": ""}
        try:
            response = requests.get("https://api.cognitive.microsoft.com/bing/v7.0/search", headers=headers, params=params)
        except Exception as e:
            print(e)
        else:
            if "statusCode" not in response.json():
                with open("/tmp/data/result/".format(domain), "w") as file:
                    file.write(json.dumps(response.json()))
                return "success"
            else:
                print("Rate limit hit, exit please")
    except Exception as ex:
        print(ex)
        return "fail for domain {}".format(domain)


def query_search_engine(url_list):
    pool = multiprocessing.pool.ThreadPool(processes=40)
    # results = set()
    for qname in pool.imap(
            worker,
            url_list,
            chunksize=1):
        # results.add(qname)
        pass
    pool.close()
    # return results
    # return response_dict


domains = []
cache = []
for file in os.listdir("/tmp/data"):
    cache.append(file.replace(".json", ""))
for file in os.listdir("/tmp/data/first_test"):
    cache.append(file.replace(".json", ""))
with open("clean.txt", "r") as file:
    for domain in file:
        domain = domain.replace("\n", "")
        if len(domains) < 120000:
            if domain not in cache:
                domains.append(domain)
        else:
            break
cache = []
query_search_engine(domains)
# test=query_search_engine(domains)
# print(len(test))
