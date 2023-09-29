#!/usr/bin/env sh

# Check whether an image called "tinyfff" exists
exists=`docker ps -a | grep tinyfff | wc -l`;
echo "Image exists: $exists";
if [ $exists -eq 0 ]; then
    echo "building the image";
    # Run docker build to build the docker file
    docker build -t tinyfff .
fi

# Run docker run to run the docker file
docker run --rm -dit \
    --name tinyfff_container \
    -v $(pwd):/home/user/tinyfff tinyfff

# Exec the program
docker exec -it tinyfff_container sh -c "cd /home/user/tinyfff && python fff_experiment_mnist.py $1 $2 $3"

docker kill tinyfff_container
