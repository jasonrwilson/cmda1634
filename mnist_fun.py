# Reads MNIST training image set and stores it as a 60000 x 784 matrix
# There are 60000 images, each of which is 28 x 28 pixels
# Each image is stored as a 28x28 = 784 dimensional row vector in the matrix
def mnist_train_set_read():
    f = gzip.open('train-images-idx3-ubyte.gz','r')
    f.read(16) # skip file header
    buf = f.read(60000*28*28)
    data = np.frombuffer(buf,dtype=np.uint8).astype(np.float32)
    train = data.reshape(60000,28*28)
    # Opening and saving the 60000 training labels
    f = gzip.open('train-labels-idx1-ubyte.gz','r')
    f.read(8) #skip header
    buf = f.read(60000)
    train_labels = np.frombuffer(buf, dtype=np.uint8)
    return train,train_labels

# Opens MNIST test image set and stores it as a 10000 x 784 matrix
# There are 10000 images, each of which is 28 x 28 pixels
# Each image is stored as a 28x28 = 784 dimensional row vector in the matrix
def mnist_test_set_read():
    f = gzip.open('t10k-images-idx3-ubyte.gz','r')
    f.read(16) # skip header
    buf = f.read(10000*28*28)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    test = data.reshape(10000,28*28)
    f = gzip.open('t10k-labels-idx1-ubyte.gz','r')
    f.read(8) #skip header
    buf = f.read(10000)
    test_labels = np.frombuffer(buf, dtype=np.uint8)
    return test,test_labels
