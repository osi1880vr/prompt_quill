# Copyright 2023 osiworx

# Licensed under the Apache License, Version 2.0 (the "License"); you
# may not use this file except in compliance with the License.  You
# may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.

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

print('wating for qdrant sucess it is up and running')

