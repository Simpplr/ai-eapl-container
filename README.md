This repository merges all Emplay code repositories together wihtin a single docker image for easy deployment.
Further, it downloads the required HuggingFace language models on the image on build stage so that it never downloads them again from the internet. This download is done by a simple [script](./download_hfmodels.py).

## To Prepare the Host:

1. Install docker first. [Here is](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/create-container-image.html#create-container-image-install-docker) a guide to do this on aws linux instance. 
2. Reboot the instance.
3. Then install docker-compose using this [guide](https://docs.docker.com/compose/install/compose-plugin/#install-the-plugin-manually)

## To Test:

1. Run `sysctl -w vm.max_map_count=262144` on the host machine as elasticsearch requires this setting to run.
2. Run `docker compose up` to start eapl and all its external requirements on your local compute.

** If you also want to test Content Recommendation endpoints,** you need to restore a mongodb. To do that;
1. Put an uncompressed folder of mongodb collections dump in [backup](./backup/) folder
2. Uncomment the `mongo-restore` service in [docker-compose](docker-compose.yaml) file. 

## To deploy:

1. Set eapl.env file for correct redis, mongodb and elasticsearch connection strings
2. Comment out the services you connected from external resources in docker-compose.yaml
3. Run `docker compose up`

