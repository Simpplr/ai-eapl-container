#!/bin/bash
#PLEASE REPLACE WITH eapl_ws with new_repo_name :%s/foo/bar/g ONLY IN VI EDITOR
#################################################################################################################################
cd `dirname "$0"`														#
export home=`pwd`														#
export python_repo_path="$1"                                                    						#
export branch="$2"														#
#################################################################################################################################
location=`ls ../ | grep "manage.py"`												#
[[ ! -z $location ]] && echo -e "\nPlease Run The Script Out Side From The eapl_ws Repository.\nExisting The App.\n" && exit 0	#
#################################################################################################################################
if [ "$python_repo_path" = "" ];then												#
   echo -e "\nplease specify the eapl_ws repo fully qualified path as first argument. e.g:"					#  
   echo -e "$0 /PATH/pbotnode\n"                                                						#
  exit 0                                                                      							#
fi                                                                              						#
#################################################################################################################################
if [ `echo $python_repo_path | rev | cut -c1` = "/" ];then									#
   echo -e "\nLast char detected as / removing the last char.\n"								#
   export python_repo_path=`echo $python_repo_path | rev | cut -c2- | rev`							#
fi																#
#################################################################################################################################
export repo_name=`echo $python_repo_path | rev | cut -d "/" -f1 | rev`								#
export python_repo_path=`echo $python_repo_path | rev | cut -d "/" -f2- | rev`							#
echo "Repository Name: $repo_name"												#
echo "Repository Path: $python_repo_path"											#
#################################################################################################################################
if [ "$branch" = "" ];then													#
	echo -e "\n No Branch Found In Second Argument!"									#
	echo -e "Deploying Master Branch!"											#
	branch="master"														#
fi																#
#################################################################################################################################
echo -ne 'START TIME ';date "+%T"
echo ""
echo ""
error_exit()
{
        echo "$1" 1>&2
        exit 1
}
cd $home && rm -rf ./eapl_ws || echo "cm repo not found" 
sudo chmod 777 /tmp && python3.6 -m venv env && . env/bin/activate || error_exit "COULD NOT ABLE TO CHANGE THE PATH"
cd $python_repo_path/$repo_name && git checkout $branch && git pull || error_exit "GIT PULL FAILED IN $repo_name"
cd $python_repo_path/$repo_name && pip3 install cython || error_exit "COULD NOT INSTALL REQUIREMENTS IN $repo_name"
cd $home && mkdir -p eapl_ws
cp -rp ${python_repo_path}/${repo_name} ${home}/eapl_ws && echo "CREATED A COPY OF $repo_name" || error_exit "COULD NOT FIND $repo_name"
rm -rf ${home}/eapl_ws/$repo_name/.git
cp ./cython.sh eapl_ws/$repo_name && echo "COPIED THE SCRIPT FILE TO $python_repo_path/$repo_name"
####################################MOVING REPO ROOT PYTHON FILES TO A DIRECTORY############################################################
mkdir -p eapl_ws/$repo_name/main_cython
find eapl_ws/$repo_name -maxdepth 1 -type f \( -iname "*.py" ! -iname "manage.py" ! -iname "rels.py" \) -exec mv -t eapl_ws/$repo_name/main_cython {} + 
echo ""
echo ""
echo "Running cython.sh File"
echo ""
echo ""
##############################################################################################################################################
bash -e eapl_ws/$repo_name/cython.sh && echo 'Successfully Encritpted all Python Files To Cython!!' || error_exit "SCRIPT.SH FAILED AND WRITE SEND MAIL SCRIPT"
cd $home
##############################################################################################################################################
mv eapl_ws/$repo_name/main_cython/*.so eapl_ws/$repo_name || echo "No Python Files To Be Encoded In Root Of The Repository"
rm -rf eapl_ws/$repo_name/main_cython
cd eapl_ws 
find . -type f \( -name "compile.py" -o -name "cython.sh" -o -name "base" \) -exec rm {} +
echo -e 'END TIME ';date "+%T"
