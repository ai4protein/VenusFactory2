import argparse
import os
import sys, errno, re, json, ssl
import urllib
from urllib import request
from urllib.error import HTTPError
from time import sleep
from tqdm import tqdm
from fake_useragent import UserAgent

ua = UserAgent()

def output_list(args):
    if args.filter_name:
        BASE_URL = f"https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/InterPro/{args.protein}/{args.filter_name}/?page_size={args.page_size}"
    else:
        BASE_URL = f"https://www.ebi.ac.uk:443/interpro/api/protein/UniProt/entry/InterPro/{args.protein}/?page_size={args.page_size}"
    print(f"Processing {BASE_URL}")

    if args.re_collect:
        os.remove(args.output)
        
    #disable SSL verification to avoid config issues
    context = ssl._create_unverified_context()
    # context.check_hostname = False
    # context.verify_mode = ssl.CERT_NONE
    
    next = BASE_URL
    attempts = 0
    cur_page = 0
    names = []
    while next:
        try:
            print(next)
            req = request.Request(next, 
                                  headers={
                                      "Accept": "application/json",
                                      'user-agent': ua.random
                                  })
            res = request.urlopen(req, context=context)
            # If the API times out due a long running query
            if res.status == 408:
                # wait just over a minute
                sleep(61)
                # then continue this loop with the same URL
                continue
            elif res.status == 204:
                #no data so leave loop
                break
            payload = json.loads(res.read().decode())
            res.close()
            next = payload["next"]
            attempts = 0
        except HTTPError as e:
            if e.code == 408:
                sleep(61)
                continue
            else:
                # If there is a different HTTP error, it wil re-try 3 times before failing
                if attempts < 3:
                    attempts += 1
                    sleep(61)
                    continue
                else:
                    sys.stderr.write("LAST URL: " + next)
                    raise e
        cur_page += 1
        bar = tqdm(payload["results"])
        for item in bar:
            bar.set_postfix({"current": f"{(cur_page - 1)*args.page_size}-{cur_page*args.page_size}"})
            names.append(item["metadata"]["accession"])
    # remove duplicate
    nemas = list(set(names))
    lenth = len(names)
    max_i = lenth//args.chunk_size+1
    for i in range(max_i):
        names_ = names[i*args.chunk_size: (i+1)*args.chunk_size]
        with open(os.path.join(args.output, f"af_raw_{args.protein_name}_{i}.txt"), "w") as f:
            for name in names_:
                f.write(name+"\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", type=str, default="IPR001557", required=False)
    parser.add_argument("--protein_name", type=str, default="MDH", required=False)
    parser.add_argument("--chunk_size", type=int, default=5000, required=False)
    parser.add_argument("--filter_name", type=str, default="", required=False)
    parser.add_argument("--page_size", type=int, default=200, required=False)
    parser.add_argument("--output", type=str, default="data/MDH", required=False)
    parser.add_argument("--re_collect", action="store_true", default=False, required=False)
    args = parser.parse_args()
    
    output_list(args)