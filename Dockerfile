FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y python3 python3-dev python3-pip && \
    apt-get install -y python3.8 python3.8-dev

RUN mkdir -m 777 /var/log/eapl_ws/
RUN touch /var/log/eapl_ws/nlp_pipeline.log /var/log/eapl_ws/request.log /var/log/eapl_ws/error.log && chmod 777 /var/log/eapl_ws/*
RUN python3.8 -m pip install pip --upgrade && python3.8 -m pip install django-environ 'spacy==2.3.0' 'sentence-transformers==0.4.1.2' pymongo freezegun
RUN python3.8 -m spacy download en_core_web_sm
RUN python3.8 -m spacy download en_core_web_md

COPY ./download_hfmodels.py /eapl_codebase/eapl_ws/
RUN python3.8 /eapl_codebase/eapl_ws/download_hfmodels.py

COPY ./eapl_codebase/ /eapl_codebase
RUN python3.8 -m pip install -r /eapl_codebase/eapl_ws/requirements.txt && python3.8 -m pip install -U -t /usr/local/lib/python3.8/dist-packages/ /eapl_codebase/eapl /eapl_codebase/eapl_nlp
COPY ./eapl.env /usr/local/lib/python3.8/dist-packages/eapl_nlp/nlp/.env
RUN cp /eapl_codebase/simpplrai/src/* /usr/local/lib/python3.8/dist-packages/eapl_nlp/nlp/

# CMD [ "python3.8", "/eapl_codebase/eapl_ws/manage.py", "runserver", "0.0.0.0:9454"]
