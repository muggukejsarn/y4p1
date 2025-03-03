import numpy as np
import imageio
import glob

# Istället för att lägga in MNIST i github:
# på stationär dator:
#         for image_path in glob.glob("D:/skola/universitet/2024_2025/p3/deeplearning/ass1/MNIST/Test/" + str(label) + "/*.png"):

# på laptopen:



def load_mnist_func(machine: str):
    
    # Loads the MNIST dataset from png images
    #
    # Return
    # X_train - Training input 
    # Y_train - Training output (one-hot encoded)
    # X_test - Test input
    # Y_test - Test output (one-hot encoded)
    #
    # Each of them uses rows as data point dimension.


    if machine == "laptop":
        print("Reading MNIST: laptop")
        testpath = "C:/skola/uni/Y4/p3/deeplearning/ass2/mnist/MNIST/MNIST/Test/"
        trainpath = "C:/skola/uni/Y4/p3/deeplearning/ass2/mnist/MNIST/MNIST/Train/"
    
    if machine == "station":
        print("Reading MNIST: stationary")
        testpath = "D:/skola/universitet/2024_2025/p3/deeplearning/ass1/MNIST/Test/"
        trainpath = "D:/skola/universitet/2024_2025/p3/deeplearning/ass1/MNIST/Train/"

 
    NUM_LABELS = 10        
    # create list of image objects
    test_images = []
    test_labels = []    
    
    for label in range(NUM_LABELS):
        print(f"In first loop label {label}")
        for image_path in glob.glob(testpath + str(label) + "/*.png"):
            image = imageio.imread(image_path)
            test_images.append(image)
            letter = [0 for _ in range(0,NUM_LABELS)]    
            letter[label] = 1
            test_labels.append(letter)  
            
    # create list of image objects
    train_images = []
    train_labels = []    
    
    for label in range(NUM_LABELS):
        for image_path in glob.glob(trainpath + str(label) + "/*.png"):
            image = imageio.imread(image_path)
            train_images.append(image)
            letter = [0 for _ in range(0,NUM_LABELS)]    
            letter[label] = 1
            train_labels.append(letter)                  
            
    X_train= np.array(train_images).reshape(-1,784)/255.0
    Y_train= np.array(train_labels)
    X_test= np.array(test_images).reshape(-1,784)/255.0
    Y_test= np.array(test_labels)
    
    return X_train, Y_train, X_test, Y_test