from typing import List, Iterable, Dict
import os
import json
import time
import requests
from itertools import chain
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("--lan", default="en", type=str)
parser.add_argument("--with_entity",  action='store_true', default=False)
args = parser.parse_args()

LANMAP = {"bn": "BN-Bangla","de": "DE-German", "en": "EN-English", "es": "ES-Spanish", "fa": "FA-Farsi", "hi": "HI-Hindi", "ko": "KO-Korean", "nl": "NL-Dutch", "ru": "RU-Russian", "tr": "TR-Turkish", "zh": "ZH-Chinese", "mix": "MIX_Code_mixed"}

VERSION = 'v1'

ES = "http://localhost:9200"

if args.lan == "mix":
    INDEX_NAME = f"*wiki_{VERSION}"
else:
    INDEX_NAME = f"{args.lan}wiki_{VERSION}"

HEADERS = {"content-type": "application/json;charset=UTF-8"}


PREFIX = f"./{VERSION}/"
LAN = args.lan


def search_batch(texts: List[str], size=10):
    url = f"{ES}/{INDEX_NAME}/_msearch"
    content = ""
    for text in texts:
        content += "{}\n"
        query = {
                "size": size,
                "query": {
                    "bool": {}
                },
                    "highlight" : {
                        "pre_tags" : ["<hit>"],
                        "post_tags" : ["</hit>"],
                        "fields" : {
                            "text" : {},
                            "title" : {}
                        }
                    }
                }
        if args.with_entity:
            query["query"]["bool"]["should"] = [{ "match": { "text":  text["sentence"] }},{ "match": { "title": {"query":text["entity"], "boost":2.0}}}]
        else:
            query["query"]["bool"]["should"] = [{ "match": { "text":  text["sentence"] }}]
        row = json.dumps(query, ensure_ascii=False)

        content += row + "\n"
    response = requests.get(url, data=content.encode("utf-8"), headers=HEADERS)
    assert response.status_code == 200
    results = list()
    for one in response.json()["responses"]:
        array = []
        for h in one['hits']['hits']:
            paragraph=h['_source']['paragraph']
            array.append((h['_source']['text'], h['_score'], paragraph, h['_source']['title'], h['_source']['url'], h['highlight']['text'] if 'text' in h['highlight'] else [''], h['highlight']['title'] if 'title' in h['highlight'] else ['']))
        results.append(array)
    return results

def _clean_space(text):
    match_regex = re.compile(u'[\u4e00-\u9fa5。，！：《》、（）]{1} +(?<![a-zA-Z])')
    should_replace_list = match_regex.findall(text)
    order_replace_list = sorted(should_replace_list,key=lambda i:len(i),reverse=True)
    for i in order_replace_list:
        if i == u' ':
            continue
        new_i = i.strip()
        text = text.replace(i,new_i)
    return text

def retrieval(part='dev', batch_size=1000):
    data = list()
    with open(f"datasets/{LANMAP[args.lan]}/{args.lan}_{part}.conll") as file:
        sentence = ""
        entity=""
        for line in file:
            line = line.strip('\n')
            if line.startswith("# id"):
                continue
            if len(line)==0:
                if len(sentence)>0:
                    if args.lan == 'zh' or args.lan == 'mix':
                        sentence = _clean_space(sentence)
                        entity = _clean_space(entity)
                    data.append([{"sentence":sentence.strip(),"entity":entity}])
                    sentence = ""
                    entity = ""
                continue
            fields = line.split()
            sentence+=fields[0]
            if fields[-1]!='O':
                if fields[-1].startswith('B-'):
                    if len(entity)==0:
                        entity+=f"{fields[0]}"
                    else:
                        entity+=f" | {fields[0]}"
                if fields[-1].startswith("I-"):
                    entity+=f"{fields[0]}"
                entity+=" "
            sentence+=" "                    
        if len(sentence)>0:
            if args.lan == 'zh' or args.lan == 'mix':
                sentence = _clean_space(sentence)
                entity = _clean_space(entity)
            data.append([{"sentence":sentence.strip(),"entity":entity}])
        print("read", len(data), "query from", file.name)

    def add_batch(batch: List, ids: List):
        timer = time.time()
        array = search_batch([o[0] for o in batch], 10)
        for o, res in zip(batch, array):
            o.append(res)
        timer = time.time() - timer
        print(ids[0], '-', ids[-1], 'seconds:', timer)
        batch.clear()
        ids.clear()

    batch, ids = list(), list()
    for i, one in enumerate(data):
        batch.append(one)
        ids.append(i)
        if len(batch) >= batch_size:
            add_batch(batch, ids)
    else:
        if len(batch) > 0:
            add_batch(batch, ids)
    with open(f"{VERSION}/{args.lan}_{part}.txt", mode='w') as writer:
        for one in data:
            writer.write(one[0]["sentence"] +'\t'+one[0]["entity"] + '\n')
            for res in one[1]:
                writer.write(res[0]+'\t'+res[2]+'\t'+res[3]+'\t'+str(res[1])+'\t'+res[4]+'\t'+res[5][0]+" ---#--- "+res[6][0]+'\n')
            else:
                writer.write('\n')
        print(writer.name)


def main():
    for p in ('dev', 'train'):
        retrieval(p)
    return


if __name__ == "__main__":
    main()

