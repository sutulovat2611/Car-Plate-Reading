# Car-Plate-Reading
The provided code is a three-layer neural network designed for classifying typical Malaysian car number plates. This neural network is specifically trained to distinguish between 10 alphabetic characters and 10 numerals, resulting in a total of 20 possible outputs.

In the initial dataset, there are 10 images for each alphabet and numeral, totaling 100 alphabetic images and 100 numeral images. The training process utilizes 80% of these images from the _"train_case"_ folder, while the remaining 20% found in the _"test_case"_ directory are reserved for testing purposes.

Moreover, the system employs a set of 10 car number plates stored in the _"target"_ folder. These plates undergo a segmentation process where individual numerals and letters are isolated before being presented to the neural network for classification. As an illustration, if the car number plate is "VBU3878," the segmentation algorithm will break it down into distinct elements: V, B, U, 3, 8, 7, and 8, and each of these characters will be classified.

## Details
In order to run the program **_A12_assignment3.py_** file is to be run, whereas the file named **_A12_report.pdf_** describes all the details regarding the project, including the methodology used, evaluation of the exeperimental results and recommendations on accuracy improvements
