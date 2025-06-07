#!/bin/bash

if [ -z "$4" ]
then
  python3 main.py --mode training --project_name $1 --run_name $2 --config $3
else
  python3 main.py --mode training --project_name $1 --run_name $2 --config $3 --checkpoint $4
fi
