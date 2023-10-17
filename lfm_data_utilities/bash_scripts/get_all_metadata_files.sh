set -e

echo "This script will search through the following folders: Uganda_full, 
Uganda_full_2, Uganda_full_3. It will copy all the perimage metadata files 
into a folder here ("all_metadata")."

echo -e "Note: this script uses 'fd', a modern version of 'find'. The HPC should already have fd installed."
mkdir -p all_metadata
fd -e csv -g "*perimage*" Uganda_full Uganda_full_2 Uganda_full_3 | xargs -I{} readlink -f {} | xargs -I{} cp {} all_metadata