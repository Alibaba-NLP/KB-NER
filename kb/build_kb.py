from typing import List, Iterable, Dict
import os
import json
import time
import requests
from itertools import chain
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lan", default="en", type=str)
args = parser.parse_args()

ES = "http://localhost:9200"
INDEX_NAME = f"{args.lan}wiki_v1"
HEADERS={'Accept': 'application/json', 'Content-type': 'application/json'}

if args.lan == "zh":
    analyzer = 'ik_max_word'
    search_analyzer = 'ik_smart'
else:
    analyzer = 'standard'
    search_analyzer = 'standard'

CONFIG = {
    "settings": {
        "number_of_shards": 1
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": analyzer, "search_analyzer": search_analyzer},
            "paragraph": {"type": "text", "index": False},
            "title": {"type": "text", "analyzer": search_analyzer, "search_analyzer": search_analyzer},
            "url": {"type": "keyword", "index":False}
        }
    }
}

PREFIX = "./dumps/"
LAN = args.lan

def batch_iter(size=10000) -> Iterable[List[Dict]]:
    batch = list()
    with open(PREFIX + f'{LAN}_all.jsonl') as file:
        for i, line in enumerate(file):
            data = json.loads(line)
            batch.append('{"index":{}}')
            batch.append(json.dumps(
                {'text': data["sentence"], "paragraph":data["paragraph"] ,'title': data["title"],'url': data['url']},
                ensure_ascii=False
            ))
            if len(batch) >= size:
                yield batch
                batch.clear()
        if len(batch)>0:
            yield batch


def add_copus():
    res = requests.put(f"{ES}/{INDEX_NAME}", json=CONFIG, headers=HEADERS)
    if res.status_code != 200:
        print(json.dumps(res.json(), indent=2))
        raise RuntimeError("failed to create index mapping!")

    def add_batch(batch: List):
        content = "\n".join(batch) + "\n"
        res = requests.post(url, data=content.encode("utf-8"), headers=HEADERS)
        if res.status_code != 200:
            failures.append(i)

    url = f"{ES}/{INDEX_NAME}/_bulk"
    batch_size = 10000
    failures = list()
    timer = time.time()
    for i, batch in enumerate(batch_iter(batch_size)):
        add_batch(batch)
        if i % 100 == 0:
            timer = time.time() - timer
            print(i, ", time", timer)
            timer = time.time()

    if len(failures) > 0:
        with open("fail", mode='w') as file:
            file.write('\n'.join(failures) + '\n')
            print('write', file.name)
    else:
        print("success")

def main():
    r = requests.get(ES, headers=HEADERS)
    assert r.status_code == 200
    add_copus()
    return


if __name__ == "__main__":
    main()

