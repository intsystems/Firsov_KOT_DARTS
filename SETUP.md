# Through Docker
  - Use my version of **Dockerfile** and **requirements.txt** to configure libraries and dependencies. This file may conflict with your version of cuda. 
  - After it make docker image  - `bash build_docker.sh`
  - If it is successfully assembled - run container - `sudo docker run --gpus=all -p 8888:8888 -v /data/:/nas/searchs -d  --name nas nas-hypernets`
  - Now you can make experiments in container - `docker exec -it nas /bin/bash`
