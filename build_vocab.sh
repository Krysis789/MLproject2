#!/bin/bash

# Note that this script uses GNU-style sed. On Mac OS, you are required to first
#    brew install gnu-sed --with-default-names
cat 'Processed_data/'$1 'Processed_data/'$2 | sed "s/ /\n/g" | grep -v "^\s*$" | sort | uniq -c > 'Processed_data/'$3
