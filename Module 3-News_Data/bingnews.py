import requests
import pprint

subscription_key = "e2c42c9a7a144d75a5c7a347055186f3"
search_term = "BJP,Modi,elections 2019"
search_url = "https://api.cognitive.microsoft.com/bing/v7.0/news/search"

headers = {"Ocp-Apim-Subscription-Key" : subscription_key,"X-Search-Location":"12.9716;long:77.5946"}
params  = {"q": search_term, "textFormat": "html","category":"Politics","mkt":"en-IN","since":"2019-01-01"}
response = requests.get(search_url, headers=headers, params=params)
response.raise_for_status()
search_results = response.json()
# pprint.pprint(search_results)
# descriptions = [article["description"] for article in search_results["value"]]

# pprint.pprint(descriptions)

with open("BJPtweets.txt","w") as f:
    results=(search_results['value'])
    
    for tw in results:
        
        print("\n")
        try:
            f.write("name: "+tw['name']+"\n"+"Date: "+tw['datePublished']+"\n"+"URL: "+tw['url'])
            f.write(tw['description']+"\n\n")
        except:
            pass
