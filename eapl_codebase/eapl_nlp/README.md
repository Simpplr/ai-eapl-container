# Directory structure:(example)


  
     1.main_dir/

          1.package_name_dir/  #eapl_nlp

               1.package_sub_dir/  #nlp


                                   1. __init__.py
                                   2.file1.py   #eapl_nlp.py
                                   3.file2.py
                                   4.filen.py
                                   
               2.__init__.py
               3.licence.txt(optional)
               4.setup.cfg(optional)
          2.README.md
          3.setup.py  




# .pypirc(append the below set of code to the file)

[distutils]

index-servers =
    gitlab


[gitlab]

repository = https://gitlab.com/api/v4/projects/20075691/packages/pypi      (project id=20075691)

username =  __ token __

password =  si4VAvyxqhxy4CaigTD1





# Commands to upload:(cd main_dir)
1.pip install .

2.python setup.py sdist  # a tar file will be created in a dist directory

3.python -m twine upload --repository gitlab /path/to/main_dir/package_name(eapl_nlp)/dist/tarfile_name.tar.gz




#When custom package is present in the install_requires in setup.py:

1.python setup.py develop --index-url https://gitlab.com/api/v4/projects/20075691/packages/pypi/simple  #dependency path

2.pip install .

3.python setup.py sdist

4.python -m twine upload --repository gitlab /path/to/main_dir/package_name(eapl_nlp)/dist/tarfile_name.tar.gz




# Installing:

pip install --extra-index-url https://__token__:si4VAvyxqhxy4CaigTD1@gitlab.com/api/v4/projects/20075691/packages/pypi/simple eapl_nlp   #package name

# Importing:

import eapl_nlp   #import package_name

# Guidelines for validation of dependent libraries
## 