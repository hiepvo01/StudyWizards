import requests

def ask_question(question, file_name):
    url = "http://127.0.0.1:5000/ask_question"
    headers = {'Content-Type': 'application/json'}
    data = {
        "question": question,
        "file_name": file_name
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Failed:", response.status_code, response.text)

if __name__ == "__main__":
    question = "What is this document about?"
    file_name = "BusinessStatistics.pdf"
    ask_question(question, file_name)
