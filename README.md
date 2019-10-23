# 3D-Reconstruction
Generate 3D image from 2D images


NOTE: WE WILL CREATE VIRTUAL ENVIRONMENT AND INSTALL ALL DEPENDENCIES IN THAT SO THAT IT WILL NOT CHANGE/AFFECT YOUR ROOT DIRECTORY, JUST BY DELETEING THIS FOLDER YOU CAN DELETE ITS EXISTANCE.

open windows 'cmd' typing cmd in search (not conda command window) as an administrator

    Download and install python from> https://www.python.org/downloads/ (if you already have then ignore this step and check its version to be sure)

    check python version> python -V

    Install virtual environment> pip install virtualenv

    test installation> virtualenv --version


    Download complete folder from GitHub as '3d_reconstruction' and change the directroy to '3d_reconstruction' folder (at this type your command prompt window should look like C:\3d_reconstruction>)

    create virtual environment repositery>virtualenv env (this will create a folder with name 'env' inside working reositery '3d_reconstruction')

    Run the command to activate virtual environment>env\Scripts\activate

(after successful run, cmd window will look like '(env) C:\3d_reconstruction>' means you have successfully created and activated virtual env and ready to work)

Install packages using (env) C:\3d_reconstruction>pip install -r requirements.txt

########## NOW YOU ARE READY RUN SCRIPT ##############################

Step 1: Calibrate Camera by issue command

(env) C:\3d_reconstruction>python calibrate.py

Step 2: Reconstruct image by following command
(env) C:\3d_reconstruction>python disparity.py

This will pop-up an intermediate image, save it to disk and press 'Q' to exit. After exit, script will generate a .ply file, visualize it on Meshalb.


References:

[1] https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_table_of_contents_calib3d/py_table_of_contents_calib3d.html

[2] https://medium.com/@omar.ps16/stereo-3d-reconstruction-with-opencv-using-an-iphone-camera-part-iii-95460d3eddf0
