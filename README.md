`Under Construction...`

# About:
This project is an implementation of a basic CNN (Convolutional Neural Network) using python. The network is already trained using CIFAR-10 dataset, and the training step is beyond the scope of this project. 

The architectural view of the network is described in the figure bellow.

![alt text](doc/cnn_schema.png)

As shown above the network is taking a an RGB image with 32x32 dimension and outputs class of the identified object in the image, 1 to 10, each number is corresponding to a label (airplane; automobile ;bird; cat etc. ).



## Technical Aspects:
- The data files that are needed for this project are located under `project_data` and can be described as the following:
    - `CNN_weigthts_3x3.txt`: which stores the convolution witghts and biases.
    - `test_batch.bin`: the test set of a 10000 image in a binary format. Each image is stored as the following:
        - The first byte is the label of the first image, which is a number in the range 0-9.
        - The next 3072 bytes are the values of the pixels of the image:
            - **First 1024 bytes** are the **red** channel values;
            - **Next 1024 bytes** are the **green** channel values; and
            - **Final 1024 bytes** are the **blue**  channel values.

- The algorithm is implemented using the following python files:
    
    - `src/images_processing.py`: which contains the functions:
        - `open_file_batch()`: which opens the `test_batch.bin` file.
        - `Binary_to_Tuple()`: which transforms the extracted data (image label, image) in the file to a tuple `(label, red_channel_32x32, green_channel_32x32, blue_channel_32x32)`.
        - `center_channel(matrice_32x32)`: which centers each channel of the image (red, blue, or green channel) from *32x32* to *24x24*.
        - `center_image(nCentered_Image)`: which centers the whole image by centering each channel using `center_channel()` function.
        - `normalise_image(centered_image)`: normalizing the image by applying a mathematical formula *(formula bellow)* which does not affect the image content, in order to be able to detecte the object in the image.

        <p align="center">
            <img src="doc/normalize_formula.png" alt="drawing" width="350">
        </p>
        
        - `Display_image(image)`: to display the image.

    - `src/convolutions.py`: which contains the function:
        - `Calc_Conv_RELU_{1, 2, or 3}(image_in, kernel_w, biases_w)`: is the function that perfomrs the convolution operation *(formula bellow)* on the feature map (the image in the first layer, and the feature map of the previous layer in the second and third layer), which takes also the kernel wights(which is fixed for the channels that are inputed to the same layer) and the biases. It also performs the RELU (REctified Linear Unit) which can be defined simply as `RELU(x)=max(0, x)`.

        <p align="center">
            <img src="doc/conv_formula.png" alt="drawing" width="350">
        </p>
        

    - `src/maxpooling.py`: which contains the function:
        - `Max_Pool(input_FM ,filter_size, stride)`: which chooses the maximum around a pixel in the window `filter_size` *(e.g 3x3)* and walking through the feature map using the parameter `stride` (e.g axb: walking a pixels in the x-axis, and b pixels in the y-axis).
    
    - `reshape_perceptron_sfmax.py`: which contains the functions:
        - `reshape_matrice_to_180(matrices_20x3x3)`: reshape the input matrices (20x3x3) to a vector of 180 elements.
        - `perceptron(I_1x180, M_180x10, Biases_1x10)`: which is performing the linear combination of all the 180 elements of the reshpaed matrices.
        - `soft_max(matrice_10x1)`: which performing the softmax operation *(formula bellow)* of the perceptron result.
            <p align="center">
                <img src="doc/softmax_formula.png" alt="drawing" width="250">
            </p>


    - `src/extracting_coeffs.py`: which contains the functions that extract *weights* and *biases* used for convolution operations and perceptron and their calling which returns the extracted data to the global variables (*matrice_conv1*; *matrice_conv2*; *matrice_conv3*; *matrice_biases_1*; *matrice_biases_2*; *matrice_biases_3*; and *matrice_biases_local*) which will be passed to their corresponding function in the main file `network_tests.py`.

    - `src/network_tests.py`: which: 
        - Contains the full implementation of the CNN (calling the function previously defined and passing the correct arguments to these functions) which presented in the figure above in *About section*. 
        - The function is mainly feeding the **test data set** *(located in `project_data/test_batch.bin` which stores 10000 32x32 RGB images and their corresponding labels)* to the CNN and check if the identified class of the object in the image (1:airplane; 2:automobile; 3:bird; 4:cat...) is corresponding to the label of the image. 
        - Then the function calculates the success rate of the processed images in realtime.
    
### Note: 
- TO-DO if there is any notes.

## Overview:
### Project State: 
- TO-DO

### To-Do List: 
- TO-DO



## Mini User Guide:
### Pre-requirements:
- Python
- Installed Numpy and Matplotlib python libraries


### How to use:
TO-DO



## Project Structure: 
    TO-DO


## Contacts:
- abderrahimelhamzi.dev@gmail.com


