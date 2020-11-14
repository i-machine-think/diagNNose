#!/usr/bin/env bash
cd scripts
sh ../test/setup.sh

python3 extract.py -c config/extract_test.json
