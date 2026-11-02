#!/usr/bin/env python3
"""One-time administrator bootstrap; passwords are prompted, never command args."""
import argparse
import getpass
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1]/'.env',override=False)
from backend.web_v2.online_access import AccessStore


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['bootstrap'])
    parser.add_argument('--email',required=True)
    args=parser.parse_args()
    password=getpass.getpass('Initial administrator password: ')
    confirmation=getpass.getpass('Confirm password: ')
    if password!=confirmation:parser.error('Passwords do not match')
    try:created=AccessStore().bootstrap(args.email,password)
    except ValueError as exc:parser.error(str(exc))
    print('Administrator created. Open /admin to sign in.' if created else 'An administrator already exists; no credentials changed.')


if __name__=='__main__':main()
