import sys
import json
import os

file = os.path.join(os.getcwd(),'installer_files/delete_after_setup',sys.argv[1])

print(f'got filename {file}')

f = open(file,'r')
data = f.read()
f.close()

json_data = json.loads(data)

print(type(json_data))

if isinstance(json_data, list):
	f = open(file,'w')
	f.write(json.dumps(json_data[0]))
	f.close()


