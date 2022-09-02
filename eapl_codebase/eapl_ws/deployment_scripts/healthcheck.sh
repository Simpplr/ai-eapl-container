#Script for health-check

#!/bin/bash
cd /home/ubuntu/eapl_ws
url=`export AWS_DEFAULT_REGION=ap-south-1; aws s3 presign s3://eaplwscodedeploymum/initscripts/warmscript.json --expires-in 3000`
python3 ./deployment_scripts/warmup.py -f $url
echo "Health Check completed"