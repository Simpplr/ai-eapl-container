
1. Copy **binary_cython** directory to outside from this repository.                
2. cd into the new binary_cython directory.                  
3. Only run the binary_script.sh and pass the fully qualified path of the eapl_ws repo Directory followed by the Branch name, we should never run **cython.sh**           

```bash
./binary_script.sh /home/ubuntu/eapl_ws/ 328-cythonize-adroit 
```
4. Once you run the script, you will find a **eapl_ws/eapl_ws** directory inside your new binary_cython folder, type below command to run the cythonize eapl_ws, make sure you are in your working python3.6 environment.          
```bash
cd ./eapl_ws/eapl_ws
python3.6 manage.py runserver
```
