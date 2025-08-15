import requests
import json

# Sample comments to test the API
sample_comments = [
    "this is an amazing and insightful video, thank you!",
    "i really did not like this, the audio was terrible.",
    "this is a tutorial about how to use python."
]

# The URL of your locally running Flask API
# Make sure the port is updated to 8081
url = "http://127.0.0.1:8081/predict"

# The data needs to be in a specific JSON format
data = {"data": sample_comments}

# Send the POST request
response = requests.post(url, json=data)

print(f"Status Code: {response.status_code}")
print("Predictions from API:")
print(response.json())