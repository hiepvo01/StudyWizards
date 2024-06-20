import requests

def upload_pdf(file_path):
    url = "http://127.0.0.1:5000/upload_pdf"
    files = {'file': open(file_path, 'rb')}
    
    response = requests.post(url, files=files)
    
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Failed:", response.status_code, response.text)

if __name__ == "__main__":
    file_path = "C:/Users/vohi0/Downloads/MathematicalModeling.pdf"
    upload_pdf(file_path)