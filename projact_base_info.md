Deep learning Model to Identify wound signs infection after surgery from images.
Master's thesis

Introduction
a sign of wound infection after surgery in the images.
In the image it is necessary to highlight (if any):
1.Suture area;
2.Edema area around the wound;
3.Hyperemia area around the wound;
4.Necrosis area;
5.Granulation area;
6.Fibrin area;
Mark for calculating the size of the wound.
The wound must be highlighted very accurately, because we need to know the area of ​​the wound in centimeters. For this, there will be a mark on the images, this is a square measuring 3x3 centimeters. Edema, necrosis, etc.
Are signs of infection in the wound.

The final playlist should be as follows.
An image is fed to the input. 
All areas of interest are automatically found on it (if they are present), they are classified. 
The area of ​​the wound is calculated at the output we receive.
    For example, in the form of json, the following information: is there swelling around the wound in the image, hyperemia, necrosis, etc. and the area of ​​the wound.

 dataset here
\data

It contains annotations in the form of polygons. You need to train a model to detect wounds. The "AllWound" label is used for this. If the file name contains "-not-," it means there is no infection. Your task is to examine the various features in the image that distinguish wounds with and without infection. You need to study textures and other features.