import argparse
import os
import ssl
from urllib import request
from lxml import etree
from tqdm import tqdm
from fake_useragent import UserAgent

ua = UserAgent()

def process(args):
    # instanciate parser
    tree = etree.parse(args.html, parser=etree.HTMLParser(recover=True))
    # get all the links
    blast_items = tree.xpath('//*[@id="root"]/div/div/div/main/div[2]/div[2]/section/div/div/span[6]/a/text()')
    context = ssl._create_unverified_context()
    
    base_url = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast/result/"
    bar = tqdm(blast_items)
    names = []
    for item in bar:
        bar.set_postfix({"current": item})
        trg_url = base_url + item + "/accs"
        req = request.Request(trg_url, 
                              headers={
                                  "Accept": "application/json",
                                  'user-agent': ua.random
                              })
        res = request.urlopen(req, context=context)
        payload = [p[5:] for p in res.read().decode().split("\n")[:-1]]
        names.extend(payload)
    # remove duplicate
    names = list(set(names))
    lenth = len(names)
    max_i = lenth//args.chunk_size+1
    for i in range(max_i):
        names_ = names[i*args.chunk_size: (i+1)*args.chunk_size]
        with open(os.path.join(args.output, f"af_raw_{args.protein_name}_{i}.txt"), "w") as f:
            for name in names_:
                f.write(name+"\n")
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein_name", type=str, default="CM", required=False)
    parser.add_argument("--html", type=str, default="data/CM/CM.html", required=False)
    parser.add_argument("--output", type=str, default="data/CM", required=False)
    parser.add_argument("--chunk_size", type=int, default=5000, required=False)
    args = parser.parse_args()
    
    process(args)