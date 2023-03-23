#!/bin/bash

# This bash script is intended for transferring a subset of folders/files from the LFM scopes' SSD drives
# locally. This script will `rsync` all the metadata files (per image and experiment files), log files,
# and folders of subsample images (~100 images per experiment at the time of writing).
# 
# The script will ask for the ngrok address and port e.g (addr=pi@4.tcp.ngrok.io, port=13843)
# and will then prompt the user to log in via ssh.
#
# An rsync dry-run will occur, listing all the folders/files that will be transferred and their total size.
# If the destination already contains folders/files that were previously transferred from that scope, `rsync` will
# skip over them (unless any changes had been made to those folders/files). This also means in the event of a connection
# interruption, you can re-initiate this script and it will pick up from it left off.

### Main process ###
trap "exit" INT # Exit whole script and not just current function on Ctrl+C
set -e # Exit script if any command fails

echo "What is the ngrok address?"
echo "(e.g pi@4.tcp.ngrok.io, exclude the port)"
read -p "> " ngrok_addr

echo "What is the port? (e.g 18434)"
read -p "> " port

echo "Where are files stored on the pi? (press enter to use the default)"
read -p "> [default:/media/pi/SamsungSSD/]: " storage_loc
storage_loc=${storage_loc:-"/media/pi/SamsungSSD/"}
[[ "${storage_loc}" != */ ]] && storage_loc="${storage_loc}/"
echo "Will search: "$storage_loc""

echo "Where do you want to store the files? (press enter to use the default)"
read -p "> [default:'.']: " local_save_loc
local_save_loc=${local_save_loc:-"."}

echo -e "DRY RUN - files/folder WILL NOT be downloaded yet: "$ngrok_addr":"$port""
echo "=========="
rsync -rzvichP --exclude ".*" --exclude '*.zip' --exclude "*zstack*" --exclude "*.zarr" --dry-run --stats -e "ssh -p $port" "$ngrok_addr":"$storage_loc" "$local_save_loc"
echo "=========="
echo -e "Transfer files? (y/n)"
read -p "> " confirmation

if [[ ! "$confirmation" =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi

echo -e "> Confirmed. Beginning transfer..."
rsync -rzvc --info=progress2 --exclude ".*" --exclude '*.zip' --exclude "*zstack*" --stats -e "ssh -p $port" "$ngrok_addr":"$storage_loc" "$local_save_loc"
