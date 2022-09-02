#!/bin/sh
server=$(grep "ServerName" /etc/apache2/sites-available/eapl_ws.conf | head -n1 | tr -s ' ' | cut -d ' ' -f3 | cut -d '.' -f1)
cd `dirname "$0"` && cd ../eapl_ws       #Changing the directory to eapl_ws
bstatus=$(git status -s | grep "^ [AM]")
if [ "$bstatus" ]
then
  echo "Cannot deploy!! Some modified or added files exist"
  echo "$bstatus"
  exit 1;
fi
git pull				#Pull the latest changes on eapl_ws
. env/bin/activate			#Activate the python virtual environment
bstatus=$(git status -s | grep "^ [AM]")
if [ "$bstatus" ]
then
  echo "Cannot deploy!! Some modified or added files exist"
  echo "$bstatus"
  exit 1;
fi
cd ../simpplrai
git pull				#Pull the latest changes on simpplrai
pip3 uninstall eapl_nlp -y
pip3 uninstall eapl -y
if [ "$username" ] && [ "$password" ]
then
 python install.py -s $server -u $username -p $password
else
 python install.py -s $server
fi
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
if [ "$server" ]
then
  echo "\n\nSERVER: $server.simpplr.services\n\nCopying files From simpplrai repo\n\n"
  cp src/*.py ../eapl_ws/env/lib/python3.8/site-packages/eapl_nlp/nlp
  echo "Files Are Copied From simpplrai/src To env/lib/python3.8/site-packages/eapl_nlp/nlp\n\n"
  cp patch/elasticsearch.py ../eapl_ws/env/lib/python3.8/site-packages/haystack/document_store
  echo "Files Are Copied From simpplrai/patch To env/lib/python3.8/site-packages/haystack/document_store\n\n"
fi
sudo supervisorctl restart eapl-ws
sudo service apache2 restart
python deployment/simpplr_warm_up.py
echo "warm_up.py run is successful"