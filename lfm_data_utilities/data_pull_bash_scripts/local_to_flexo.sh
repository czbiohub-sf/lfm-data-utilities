#!/bin/bash

trap "exit" INT # Exit whole script and not just current function on Ctrl+C
set -e # Exit script if any command fails

function confirmation() {
    if [[ ! $1 =~ ^[Yy]$ ]]
then
    [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1 # handle exits from shell or function but don't exit interactive shell
fi
}

echo "What is your user name (first.last)"
read -p "> " user_name

echo "Which Biohub ssh node? (press enter to use the default)"
read -p "> [default:login-01.czbiohub.org]: " login_node
login_node=${login_node:-"login-01.czbiohub.org"}
echo "Login node: "$login_node""

echo "FROM | Where are the files stored? (press enter to use the default)"
read -p "> [default:/media/pi/SamsungSSD/]: " storage_loc
storage_loc=${storage_loc:-"/media/pi/SamsungSSD/"}
[[ "${storage_loc}" != */ ]] && storage_loc="${storage_loc}/"
echo "Will send: "$storage_loc""


echo "TO | Scope name (lowercase)?"
read -p "> [curiosity/insight/phoenix]: " save_loc
save_loc="/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_Scope/Uganda_Subsample/$save_loc"
echo "Saving to "$save_loc""

echo -e "\nPARAMS"
echo "- user_name: "$user_name""
echo "- node: "$login_node""
echo "- pi storage location: "$storage_loc""
echo "- remote storage location: "$save_loc""
echo -e "\n"
echo "Confirm parameters? (y/n)"
read -p "> " confirm
confirmation "$confirm"

echo ""$user_name"@$login_node":\"""$save_loc"""\""

echo -e "> DRY RUN - files/folder WILL NOT be sent yet"
echo "=========="
rsync -rzvichP --exclude ".*" --exclude '*.zip' --exclude "*zstack*" --exclude "*.zarr" --dry-run --stats "$storage_loc" "$user_name"@"$login_node":"$save_loc"
echo "=========="
echo -e "> Transfer files? (y/n)"
read -p "> " confirm

confirmation "$confirm"

echo -e "> Confirmed. Beginning transfer..."
rsync -rzvc --info=progress2 --exclude ".*" --exclude '*.zip' --exclude "*zstack*" --exclude "*.zarr" --stats "$storage_loc" "$user_name"@"$login_node":"$save_loc"
