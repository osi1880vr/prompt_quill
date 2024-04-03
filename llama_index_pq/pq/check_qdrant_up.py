import requests
import time



trycount = 0
while 1:
    print('wating for qdrant to get up and running')
    # The API endpoint
    url = "http://localhost:6333/dashboard"

    # A GET request to the API
    try:
        response = requests.get(url)
        if response.status_code == 200:
            break
        time.sleep(1)
    except:
        pass
    if trycount > 300:
        print('something went wrong qdrant is still not up and running, please try again')
        break
    trycount += 1



