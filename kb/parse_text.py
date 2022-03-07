from wikipedia2vec.dump_db import DumpDB
import json
import argparse
import nltk.data
from nltk.tokenize import RegexpTokenizer
from spacy.lang.en import English

nlp = English()
config = {"punct_chars": None}
nlp.add_pipe("sentencizer", config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--paragraph", default="all", type=str)
parser.add_argument("--lan", default="en", type=str)
args = parser.parse_args()

LANMAP = {"en": "english", "de": "german", "ru": "russian", "es": "spanish", "nl": "dutch", "tr": "turkish"}
tokenizer = None
if args.lan in LANMAP:
    tokenizer = nltk.data.load(f"tokenizers/punkt/{LANMAP[args.lan]}.pickle")
elif args.lan == "zh":
    tokenizer = RegexpTokenizer(".*?[。！？]")

def split_sent(text):
    if tokenizer is not None:
        return tokenizer.tokenize(text)
    else:
        doc = nlp(text)
        sents_list = []
        for sent in doc.sents:
           sents_list.append(sent.text)
        return sents_list

db = DumpDB(f"dumps/{args.lan}.out")
wf = open(f"dumps/{args.lan}_{args.paragraph}.jsonl", "w")
p_id = 0

for title in db.titles():
    unique = set()
    paragraphs = db.get_paragraphs(title)
    if len(paragraphs) == 0:
        continue
    if args.paragraph == "first":
        paragraphs = paragraphs[:1]
    for p_id, paragraph in enumerate(paragraphs):
        text = paragraph.text.strip()
        offset = paragraphs[p_id].text.index(text)
        wiki_links = paragraphs[p_id].wiki_links
        if text in unique:
            # print(f"repeat -->  {text.encode('utf-8', 'replace').decode('utf-8')}")
            continue
        unique.add(text)
        text_with_entity = (text + '.')[:-1]
        wiki_links = sorted(wiki_links, key= lambda x:x.start, reverse=True)
        for anchor in wiki_links:
            char_start = anchor.start - offset # 由于strip，偏移空白字符
            char_end = anchor.end - offset  # 由于strip，偏移空白字符
            mention = anchor.text
            entity = anchor.title.encode('utf-8', errors='replace').decode('utf-8')
            try:
                entity = db.resolve_redirect(anchor.title)
            except Exception as e:
                print(e)
            if char_end >= len(text):
                char_end = len(text)
            if char_start < 0:
                char_start = 0
            if char_start>len(text):
                char_start = len(text)-1
            text_with_entity = text_with_entity[:char_end] + f'</e>' + text_with_entity[char_end:]
            text_with_entity = text_with_entity[:char_start] + f'<e:{entity}>' + text_with_entity[char_start:]


        sentences = []
        try:
            sentences = split_sent(text)
        except Exception as e:
            print(e)
        s_id = 0
        for _, sentence in enumerate(sentences):
            example = {"title": title, "p_id":p_id, "s_id":s_id, "sentence": sentence, "paragraph": text_with_entity, "url": f"https://{args.lan}.wikipedia.org/wiki/{title}"}
            try:
                wf.write(json.dumps(example, ensure_ascii=False).encode('utf-8', 'replace').decode('utf-8') + "\n")
                s_id+=1
            except Exception as e:
                print(e)

wf.close()
