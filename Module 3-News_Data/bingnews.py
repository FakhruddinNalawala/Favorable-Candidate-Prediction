import requests
import pprint

subscription_key = "ef5ae340485646f6a235a8c1c83ee277"
search_term = "BJP,Modi,Narenda Modi,Rajnath Singh,Amit,Shah,bhartiya janata party"
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/news/search"

headers = {"Ocp-Apim-Subscription-Key" : subscription_key,"X-Search-Location":"12.9716;long:77.5946"}
params  = {"q": search_term, "textFormat": "html","category":"Politics","mkt":"en-IN","since":"2018-01-01"}
response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
search_results = response.json()
# pprint.pprint(search_results)
# descriptions = [article["description"] for article in search_results["value"]]

# pprint.pprint(descriptions)

with open("BJP_bing_news.txt","w") as f:
    results=(search_results['value'])
    
    for tw in results:
        
        try:
            # f.write("name: "+tw['name']+"\n"+"Date: "+tw['datePublished']+"\n"+"URL: "+tw['url'])
            f.write(tw['description']+"\n")
        except:
            pass

search_term = "INC,Congress,Rahul Gandhi,indian national congress,sonia gandhi,Manmohan Singh"
params  = {"q": search_term, "textFormat": "html","category":"Politics","mkt":"en-IN","since":"2018-01-01"}
response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
search_results = response.json()
# pprint.pprint(search_results)
# descriptions = [article["description"] for article in search_results["value"]]

# pprint.pprint(descriptions)

with open("INC_bing_news.txt","w") as f:
    results=(search_results['value'])
    
    for tw in results:
        
        try:
            # f.write("name: "+tw['name']+"\n"+"Date: "+tw['datePublished']+"\n"+"URL: "+tw['url'])
            f.write(tw['description']+"\n")
        except:
            pass
