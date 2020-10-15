import requests

BASE = "http://0.0.0.0:5000/"

response = requests.get(BASE + "classify/2")

print(response.json())
