import sys
import argparse

from network_tests import testing_network

# Global parameter
image_size = 3073 # in bytes



parser = argparse.ArgumentParser(
                                prog= "CNN_TEST_PROG",
                                description = "Main program for launching test for the CNN implementation" 
                                )

parser.add_argument("--data_dir", type=str, help="Directory path of 'test data set' and 'Coeffs'.")
parser.add_argument("--image_iter", type=int, help="Number of images to iterate in the 'test data set'")
args = parser.parse_args()

parser.print_help()

if(args.data_dir == None):
    data_set_test_file = "./project_data/test_batch.bin"
    coeffs_file        = "./project_data/CNN_coeffs_3x3.txt"
elif (args.data_dir != None):
    data_set_test_file = args.data_dir + "/test_batch.bin"
    coeffs_file        = args.data_dir + "/CNN_coeffs_3x3.txt"

if(args.image_iter == None):
    image_iter = 10
elif (args.image_iter != None):
    image_iter = args.image_iter


# Run the test of the CNN network:
testing_network(image_iter, data_set_test_file, coeffs_file, image_size)

