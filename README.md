
# Table of Contents

1.  [Antarstick](#org5f95da3)
    1.  [What is Antarstick](#org7aba648)
    2.  [Features](#orgd78e9ac)
    3.  [Installation](#org233d2ed)
    4.  [User guide](#org62d4578)


<a id="org5f95da3"></a>

# Antarstick


<a id="org7aba648"></a>

## What is Antarstick

Antarstick is a program for analyzing height of snow in photos. The program processes time-lapse
photos from a fixed camera capturing wooden stakes with a known length. In each photo the stakes are
used as a reference to estimate the snow height at the place of each stake.


<a id="orgd78e9ac"></a>

## Features

-   simple graphical user interface
-   automatic suggestion/detection of stakes
-   stake editing
-   option to link cameras that capture scene from different point of view potentially
    resulting in one stake being captured multiple times
-   automatic stake aligning after small camera movements
-   semi-automatic stake aligning after significant camera movements


<a id="org233d2ed"></a>

## Pre-requisities
1. Install Qt5
   
    Windows: download from https://www.qt.io/download-qt-installer
   
    Ubuntu-based OS: `sudo apt-get install qt5-default`
    
2. Install OpenCV
    Windows: download `opencv-4.2.0-vc14_vc15.exe` from https://github.com/opencv/opencv/releases/download/4.2.0/opencv-4.2.0-vc14_vc15.exe
    
    For Linux, please refer to the OpenCV installation guide at: https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html
    and replace occurrences of `https://github.com/opencv/opencv/archive/master.zip` with `https://github.com/opencv/opencv/archive/4.2.0.zip`

## Installation

1.  Clone this repo by running
    
        git clone https://github.com/mrazr/antarstick
    
    or download and extract the zip archive by clicking Code -> Download ZIP at the upper part of the page

2.  Navigate terminal/command line to the root directory(antarstick) of the project

3.  Install requirements
    1.  On Linux it should suffice to run the script antarstick\_install\_linux.sh
    
    2.  Windows:
        1.  Create a new python virtual environment:
            
                python -m venv venv
        
        2.  Install requirements:
            
                ./venv/Scripts/pip3.exe install -r requirements.txt
            
            If you use command line instead of powershell/windows 10 terminal then substitute forward slashes
            with backward slashes


<a id="org62d4578"></a>

## Brief user guide

1.  Still inside the root directory, run the application by executing
    
        ./venv/bin/python3 mainwindow.py
    
    in terminal on Linux, or
    
        ./venv/Scripts/python.exe mainwindow.py
    
    in Powershell on Windows.
    

2.  You&rsquo;ll be presented with options to open a camera or a dataset. Press **Add camera** and choose a folder
    that contains photos.
3.  On the left side of the application you&rsquo;ll find a list of photo names and on the right side, occupying
    most of the application is the photo viewer. By clicking on the name of an image, the image will be
    displayed.
4.  Find a photo with no snow, all sticks clearly visible, from top to bottom. The image should not be too bright nor too dim.
5.  Press **Find sticks** in the top menu inside the photo viewer.
    1.  Ideally, all stakes should be graphically marked with a green frame and circles with cross laying on top of each stake&rsquo;s extremities.
    2.  If some stakes are misaligned or you want to delete undesired stakes/stakes that are not really stakes,
        press **Edit sticks** in the top menu.
        1.  Press the red **x** to delete a stake.
        2.  Drag the circular handles to either move top/bottom endpoints or the middle to move the whole line.
        3.  When you&rsquo;re done exit the **Edit mode** by clicking the **Edit sticks** button.
    3.  To ease the stick editing, you can zoom by scrolling and pan the view either by pressing the **Middle mouse button** and dragging or by simultaneously pressing **Ctrl** and **Left mouse button** and dragging.
6.  Press **Confirm sticks**.
7.  Press **Process photos**
    
    A menu will appear, where you&rsquo;ll have an option to choose how many processes you want to spawn.
    Considering the analysis procedure is relatively computationally demanding, bear in mind that choosing a number of processes that matches the number of processes/cores of your CPU will have an effect on the responsiveness of the whole system.
    
    For now, due to the way Windows creates processes and handles their memory, the Windows version is limited to 2
    processes per camera to avoid huge consumption of RAM.
8. The **Measurement mode** allows you to adjust incorrect measurements. Incorrect measurements can be corrected by clicking at the appropriate level in the sticks that have been assigned erroneous measurements.


