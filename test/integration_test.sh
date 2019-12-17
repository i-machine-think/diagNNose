#!/usr/bin/env bash
cd scripts
sh test/setup.sh

python3 extract.py -c extract_test.json
