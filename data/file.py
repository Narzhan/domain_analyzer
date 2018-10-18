import requests

api_key = "AIzaSyAYgHJranA_33jTS5PBV6crFLAEYyX6VEA"
search = "003566780843716268693:nkb-lxqucd8"
# string = "picpica.net"
# for string in search_strings:
#     try:
#         res = requests.get(
#             "https://www.googleapis.com/customsearch/v1?key={}&cx={}&q={}".format(api_key, search, string))
#     except Exception as e:
#         print(e)
#     else:
#         with open("black/{}.json".format(string), "w") as file:
#             file.write(res.text)

import requests, json, time
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/search"
headers = {"Ocp-Apim-Subscription-Key": "0192313657c949839f025d9d6f027d17"}
i=0
# for k in ["google.com", "fecabook.com", "apple.com", "microsoft.com", "seznam.cz", "blizzfan.cz", "wikipedia.cz"]:
# with open("blacklist.txt", "r") as file:
#     for domain in file:
#         domain=domain.replace("\n", "")
domain = "amerex-gastro.cz"
params = {"q": "{}+site:{}".format(domain, domain), "textDecorations": True, "textFormat": "HTML"}
response = requests.get(search_url, headers=headers, params=params)
# response = requests.Request("GET",search_url, headers=headers, params=params)
# prep= response.prepare()
# print(prep.url)
response.raise_for_status()
search_results = response.json()
print(search_results)
# with open("D:\\Narzhan\\Documents\\dipl\\data\\black\\{}.json".format(domain), "w") as file:
#     file.write(json.dumps(search_results))
        # i+=1
        # if i == 50:
        #     time.sleep(5)
        #     i=0
    # print(search_results)
