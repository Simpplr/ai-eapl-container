#Script for installing packages and restarting services

#!/bin/bash
cd /home/ubuntu/eapl_ws
. env/bin/activate
git checkout master
git pull
pip3 uninstall eapl_nlp -y
pip3 uninstall eapl -y
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
sudo service apache2 restart
sudo supervisorctl restart all
url=`export AWS_DEFAULT_REGION=ap-south-1; aws s3 presign s3://eaplwscodedeploymum/initscripts/warmscript.json --expires-in 3000`
python3 ./deployment_scripts/warmup.py -f $url
echo "ALL DONE"