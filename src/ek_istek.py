import json
with open("C:/turtle_project/data/raw/annotations.json") as f:
    data = json.load(f)
    print(type(data))          # list mi dict mi?
    
    if isinstance(data, list):
        print(data[0])         # ilk elemanın tipi ve içeriği
    else:
        print(list(data.keys())[:5])   # sözlükse anahtarları
    print(data["images"][0])