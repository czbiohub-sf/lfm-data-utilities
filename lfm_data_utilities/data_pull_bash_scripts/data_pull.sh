#!/bin/bash
# Pull data from Remoscopes using ngrok
# Requires rsync



trap "echo 'Script interrupted by user, exiting.'; exit" INT
set -e

# Global variables
ngrok_addr=""
port=""
storage_loc=""
local_save_loc=""
control_path=""

intro_message() {
    printf "\033[1;35m==========================\033[0m\n"
    printf "\033[1;34mRemoscope Data Pull Script\033[0m\n"
    printf "\033[1;35m==========================\033[0m\n"
    echo "REQUIREMENTS:"
    echo "1. You will need rsync v3.1.3. (other versions might work but have not been tested)"
    echo "2. You should be on BRUNO while running this script. Certain UNIX commands such as 'mapfile' may not be present on your local computer."
    echo "3. The technicians need to run 'ng' on the scopes there for this script to be able to establish a connection."
    echo "4. Get the ngrok address and port number from the Remoscope builds sheet, you will be asked for them shortly."
    printf "\033[1;35m==========================\033[0m\n\n"
}

setup_ssh() {
    # Get and store ngrok address and port number
    read -p "What is the ngrok address? (e.g. pi@4.tcp.ngrok.io, exclude the port) > " ngrok_addr
    read -p "What is the port? (e.g. 18434) > " port
}

setup_paths() {
    # Get and store remote directory to pull from (defaults to /media/pi/SamsungSSD) and where to save (defaults to current directory)
    read -p "Where are files stored on the pi? [default: /media/pi/SamsungSSD/] > " storage_loc
    storage_loc=${storage_loc:-"/media/pi/SamsungSSD/"}
    [[ "${storage_loc}" != */ ]] && storage_loc="${storage_loc}/"
    echo "Remote storage location set to: $storage_loc"

    read -p "Where do you want to store the files locally? [default: .] > " local_save_loc
    local_save_loc=${local_save_loc:-"."}
}

setup_ssh_connection() {
    # Use ControlPath to avoid the headache of re-entering the password repeatedly
    control_path="/tmp/ssh_mux_${ngrok_addr//@/_}_${port}"
    echo "Establishing persistent SSH connection..."
    ssh -MNf -o ControlPath="${control_path}" -o ControlPersist=5400 -p "${port}" "${ngrok_addr}" || {
        echo "Failed to establish persistent SSH connection."
        echo ""
        echo -e "\033[1;33m###############################################\033[0m"
        echo -e "\033[1;31mPlease verify the following: \033[0m"
        echo -e "\033[1;31m  1. Check that the ngrok address is correct: addr=${ngrok_addr} port=${port}\033[0m"
        echo -e "\033[1;31m  2. Check that the tunnel is active - did the technicians run 'ng' on the scope?\033[0m"
        echo -e "\033[1;33m###############################################\033[0m"
        echo ""
        exit 1
    }
}

# Checks if the persistent connection is active; if not, re-establish it.
ensure_control_connection() {
    if ! ssh -O check -o ControlPath="${control_path}" "${ngrok_addr}" 2>/dev/null; then
         echo "SSH master connection lost. Re-establishing..."
         setup_ssh_connection
    fi
}

cleanup() {
    echo "Closing persistent SSH connection..."
    ssh -O exit -o ControlPath="${control_path}" "${ngrok_addr}" 2>/dev/null || true
}
trap cleanup EXIT

sync_files() {

    # Show user the run folders by date
    echo "Available remote folders:"
    ssh -p "${port}" -o ControlPath="${control_path}" "${ngrok_addr}" "find ${storage_loc} -mindepth 1 -maxdepth 1 -type d"

    read -p "Enter a starting date for data collection (YYYY-MM-DD) > " date

    # Get all folders including and after the specified date
    mapfile -t dirs < <(ssh -p "${port}" -o ControlPath="${control_path}" "${ngrok_addr}" \
        "find ${storage_loc} -mindepth 1 -maxdepth 1 -type d -newermt '${date}'")

    if [ ${#dirs[@]} -eq 0 ]; then
        echo "No directories found newer than ${date}."
        exit 0
    fi

    # Sync each directory individually with a retry loop
    # At the moment, if the connection completely drops on their end, this will still require the user to re-enter the
    # password.
    for remote_dir in "${dirs[@]}"; do
        echo "Syncing ${remote_dir}..."
        while true; do
            ensure_control_connection
            if rsync -rzuv --info=progress2 \
                --exclude ".*" --exclude "*.zip" --exclude "*zstack*" --exclude "*.zarr" --exclude "*.npy" \
                --stats -e "ssh -p ${port} -o ControlPath=${control_path}" \
                "${ngrok_addr}:${remote_dir}" "${local_save_loc}"; then
                break
            else
                echo "Rsync failed for ${remote_dir}. Retrying in 10 seconds..."
                sleep 10
            fi
        done
    done
}

intro_message
setup_ssh
setup_paths
setup_ssh_connection
sync_files

