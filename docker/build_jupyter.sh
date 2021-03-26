#!/bin/bash
docker build -f jupyter-dockerfile-ubuntu1804 \
             -t gruber/nn-decoding-jupyter .
