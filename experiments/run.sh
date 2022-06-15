#!/bin/bash

export SCENARIO='scenario-9-sbrt-timed-08M'

cd ./$SCENARIO/by-epoch && python DL-CD-MAP-GPU-EP.py | tee DL-CD-MAP-GPU-EP.log
cd -
cd ./$SCENARIO/by-topology-4-layers && python DL-CD-MAP-GPU-4L.py | tee DL-CD-MAP-GPU-4L.log
cd -
cd ./$SCENARIO/by-topology-5-layers && python DL-CD-MAP-GPU-5L.py | tee DL-CD-MAP-GPU-5L.log
cd -
cd ./$SCENARIO/by-topology-6-layers && python DL-CD-MAP-GPU-6L.py | tee DL-CD-MAP-GPU-6L.log
