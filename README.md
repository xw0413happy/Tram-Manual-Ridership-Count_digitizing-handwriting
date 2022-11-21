
# Image Cropping and digitize handwriting


The beginning idea was to convert each scanned pdf into png, and then use machine learning analytics to digitize total ridership number (by using *cropping*, please look at .py codes for more details) from the images (.png). However, The idea was supspended due to the following reasons.
- (1) when using cropping image, scanned pdf can not make sure that each cropped area will exactly locate us to "total ridership"
- (2) even though python can recognize all numbers on images, due to less than 100% accuracy machine learning results, we couldn't find which numbers recognized by machine learnings are not correct ridership numbers.


## File Format

This project is used by the following file formats:

- **training & testing csv files**, which is downloaded from internet (please find yourself)
- **.png image files**, one way is writing on screen, another way to crop a specific area (to locate *total ridership*) from tram sheets


## Deployment

To deploy this project run, the following modules are needed to be imported as belows.

```bash
import os
import pytesseract
from pytesseract import Output
import re
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import keras
from keras.datasets import mnist # import data
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from itertools import groupby
```


## Repository Structure


| File Name | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `digitizing handwriting` | `.py` | **Required**. It is an imcomplete file. |

#### Other supplementary files description

```http
All uploaded .png files are used for testing.
```


## Author

- [@xw0413happy](https://github.com/xw0413happy)


## 🚀 About Me
I took 2 python classes during my M.S. degree-seeking program, now I am a computer language amateur, strong desire to learn more.


## 🛠 Skills
Python, R, SQL, ArcGIS, Javascript, HTML, CSS, Nlogit, Synchro, Vissim, AutoCAD, Stata


## Acknowledgements

 - [LeeTran](https://www.leegov.com/leetran/how-to-ride/maps-schedules)
 - [Learn more about how to loop over images by using Python Tkinter](https://www.youtube.com/watch?v=NoTM8JciWaQ&t=565s)
 - [Genfare](https://www.genfare.com/products/)

