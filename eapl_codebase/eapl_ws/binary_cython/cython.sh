#!/bin/bash
cd `dirname "$0"`
pwds=`pwd`
: '
from setuptools import setup
from setuptools.extension import Extension

from Cython.Build import cythonize
from Cython.Distutils import build_ext

setup(
  name=value,
  ext_modules=cythonize(
   [
     Extension(value+".*", [value+"/*.py"]),
   ],
   build_dir="build",
   compiler_directives={'language_level' : "3"}),
  cmdclass=dict(
   build_ext=build_ext
  ),
  packages=[value]
)
'
sed -n '5,23p' ./cython.sh > ../base			 #CREATING A FILE NAMED "base FROM LINE 5th to 25th LINE
folders=`find -type f -name "*.py" | grep -v "__init__.py" | rev | cut -d "/" -f2- | rev | sort | uniq | tail -n +2 | tr '\r\n' ' '`
cython () {
cd $pwds                                              	 #CHANGING DIRECTORY TO HERE (where ever the script is)
rm -rf /tmp/cython           			   	 #REMOVING OLD CYTHON FILES
cp ../base ../compile.py        			 #COPING base FILE TO compile.py
sed -i "1s/^/value = '$value'\n/" ../compile.py          #ADDING value = '' into compile.py
cp ../compile.py $base && cd $base                       #COPING compile.py FILE to parent directory and cd into the same
python3 compile.py build_ext --build-lib /tmp/cython     #CYTHONINZING
sleep 2							 #SLEEPING FOR 2 SEC BEACUSE WE NEED TO WAIT FOR .SO FILES
mv `find /tmp/cython/ -iname "*.so"` $value              #COPING THE SO FILES TO REQUIRED DIRECTORY
rm -rf $value/*.py					 #REMOVING THAT PYTHON FILE
echo -e "\n\nCythonization Done For \t $base/$value\n\n\t\t\t\t<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>"
}

#For Loop through an array of strings in bash script

for FolderName in `echo $folders`; do
  value=`echo $FolderName | rev | cut -d "/" -f1 | rev`
  base=`echo $FolderName | rev | cut -d "/" -f2- | rev`
  echo -e "\n\nCythonic Process Started For : \t $FolderName\nValue is: \t $value\nBase is: \t $base\npwd: `pwd`\n`ls -l` \n\n"
  cython
done
