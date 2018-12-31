#!/usr/bin/env bash

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rZufpopTpqjeMNGD1bHgORTw7WF4mmug' -O train.zip

# wget --quiet --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rZufpopTpqjeMNGD1bHgORTw7WF4mmug' -O train.zip
unzip -o -q train.zip
ls train | head