set -e

echo "USAGE"
echo -e "=================================="
echo "./get_all_metadata_files.sh path_to_save"
echo "e.g: ./get_all_metadata_files.sh /hpc/mydata/first.last"
echo "=================================="

if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please provide a path to save the folder, e.g "/hpc/mydata/first.last""
    exit 1
fi


TLD="/hpc/projects/group.bioengineering/LFM_scope"
echo -e "\nThis script will search through the following folders: Uganda_full, Uganda_full_2, Uganda_full_3. It will copy all the perimage metadata files into a folder called "all_metadata" in the location you specified."
mkdir -p $1/all_metadata
fd -e csv -g "*perimage*" $TLD/Uganda_full $TLD/Uganda_full_2 $TLD/Uganda_full_3 | xargs -I{} readlink -f {} | xargs -I{} cp {} $1/all_metadata