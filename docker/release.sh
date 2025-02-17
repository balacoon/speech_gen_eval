#!/bin/bash 
# Copyright 2025 Balacoon

set -e
set -x

usage() {
    cat <<EOF
Usage:
bash $1 --username <docker_username> --password <docker_password>
EOF
}

script_dir="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
version=$(cat "$script_dir/VERSION")

username=""
password=""
while [ "$1" != "" ]; do
    case $1 in
        -h | --help ) usage $0
            exit
            ;;
        --username ) shift
            username=$1
            ;;
        --password ) shift
            password=$1
            ;;
        * ) usage $0
            exit 1
    esac
    shift
done

if [ -z "$username" ] || [ -z "$password" ]; then
    echo "Error: Docker username and password are required"
    usage $0
    exit 1
fi

if [[ "$(docker images -q speech_gen_eval:latest 2> /dev/null)" == "" ]]; then
    echo "There is no speech_gen_eval image, build it with docker/build.sh"
    exit 1 
fi

# Check if remote docker exists
docker login -u "$username" -p "$password"
if [[ $(docker manifest inspect balacoon/speech_gen_eval:$version > /dev/null ; echo $?) -eq 0 ]]; then
    echo "balacoon/speech_gen_eval:$version already exist, consider bumping the version"
    exit 1
fi
# Squash local built into release image
docker-squash -t balacoon/speech_gen_eval:$version speech_gen_eval:latest

# push image to the docker hub
docker push balacoon/speech_gen_eval:$version
