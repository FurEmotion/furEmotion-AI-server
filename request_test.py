import requests

url = 'http://localhost:7230/'

user_id = 'your_user_id'
wav_file_path = 'test_data/hostile.wav'

files = {'file': open(wav_file_path, 'rb')}
data = {'user_id': user_id, 'species': 'dog'}

response = requests.post(url, files=files, data=data)

print(f"Status Code: {response.status_code}")
print(f"Response JSON: {response.json()}")
