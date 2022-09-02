#!/bin/sh
cd `dirname "$0"`
#curbranch=$(git rev-parse --abbrev-ref HEAD)
#if [ "$curbranch" != "master" ]
#then
#  echo "Current branch is:" "$curbranch"
#  echo "Cannot deploy!! Current branch is not master."
#  echo "Please switch to master and run this script again."
#  exit 1;
#fi
bstatus=$(git status -s | grep "^ [AM]")
if [ "$bstatus" ]
then
  echo "Cannot deploy!! Some modified or added files exist"
  echo "$bstatus"
  exit 1;
fi
git pull
. env/bin/activate
pip3 uninstall eapl_nlp -y
pip3 uninstall eapl -y
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
sudo supervisorctl restart eapl-ws
sudo service apache2 restart
