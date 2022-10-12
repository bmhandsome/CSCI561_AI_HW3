import sys
import numpy
import time

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]

TRAIN_IMAGE_PATH = arg1
TRAIN_LABEL_PATH = arg2
TEST_IMAGE_PATH = arg3
TEST_OUTPUT_PATH = "test_predictions.csv"

MAX_TIME_FOR_TRAINING = 25 * 60 * 1000 # 25 Mins

def getTimeNowInMilli():
    return int(time.time() * 1000)

START_TIME = getTimeNowInMilli()
print(f"START_TIME: {START_TIME}")

NUM_INPUT = 784
NUM_OUTPUT = 10
BATCH_SIZE = 256
num_hidden1 = 512
num_hidden2 = 256
LEARNING_RATE = 0.001
EPOCHS = 400


def read_file(file_path):
    return numpy.genfromtxt(file_path, delimiter=",")

def write_to_file(file_path, data):
    return numpy.savetxt(file_path, data, fmt="%d")

train_images = read_file(TRAIN_IMAGE_PATH)
train_labels = numpy.array(read_file(TRAIN_LABEL_PATH)).reshape(-1, 1)
train_labels_new_format = numpy.zeros((train_labels.shape[0], 10))

#convert format of train_labels
for i in range(len(train_labels)): 
    answer = train_labels[i][0]
    new_format_array = numpy.zeros(10)
    new_format_array[int(answer)] = 1
    train_labels_new_format[i] = new_format_array

weight1 = numpy.random.normal(loc = 0.0, scale=0.01, size=(NUM_INPUT, num_hidden1))
weight2 = numpy.random.normal(loc = 0.0, scale=0.01, size=(num_hidden1, num_hidden2))
weight3 = numpy.random.normal(loc = 0.0, scale=0.01, size=(num_hidden2, NUM_OUTPUT))

bias1 = numpy.zeros(num_hidden1)
bias2 = numpy.zeros(num_hidden2)
bias3 = numpy.zeros(NUM_OUTPUT)

derivative_weight1 = numpy.zeros((NUM_INPUT, num_hidden1))
derivative_weight2 = numpy.zeros((num_hidden1, num_hidden2))
derivative_weight3 = numpy.zeros((num_hidden2, NUM_OUTPUT))

derivative_bias1 = numpy.zeros(num_hidden1)
derivative_bias2 = numpy.zeros(num_hidden2)
derivative_bias3 = numpy.zeros(NUM_OUTPUT)

activation0 = numpy.zeros(NUM_INPUT)
activation1 = numpy.zeros(num_hidden1)
activation2 = numpy.zeros(num_hidden2)
activation3 = numpy.zeros(NUM_OUTPUT)

z1 = numpy.zeros(num_hidden1)
z2 = numpy.zeros(num_hidden2)
z3 = numpy.zeros(NUM_OUTPUT)

def activation_function_sigmoid(x): 
    x = numpy.clip(x, -700, 700 )
    z = numpy.exp(-x)
    return 1 / (1 + z)


def derivative_sigmoid(x):
# x is activation
    return x * (1.0 - x)

def softmax(x):
    return numpy.exp(x) / numpy.sum(numpy.exp(x), axis=1).reshape(x.shape[0], 1)

def forward_pass(inputs):
    global activation0
    global activation1
    global activation2
    global activation3
    global z1
    global z2
    global z3

    activation0 = inputs
  
    z1 = numpy.dot(activation0, weight1) + bias1  
    activation1 = activation_function_sigmoid(z1)

    z2 = numpy.dot(activation1, weight2) + bias2    
    activation2 = activation_function_sigmoid(z2)

    z3 = numpy.dot(activation2, weight3)   
    activation3 = softmax(z3)
    return activation3

def loss_function(real_answers):
    num_input = real_answers.shape[0]
    sum_cross_entropy = numpy.sum(real_answers * numpy.log(activation3))
    return (-1) * sum_cross_entropy / num_input

def backward_propagation(real_answers):
    error = activation3 - real_answers
    global derivative_weight3 
    global derivative_weight2 
    global derivative_weight1

    global derivative_bias3
    global derivative_bias2
    global derivative_bias1

    derivative_weight3 = numpy.dot(activation2.T, error)
    derivative_bias3 = numpy.sum(error, axis=0)

    derivative_z2 = numpy.dot(error, weight3.T) * derivative_sigmoid(activation2)
    derivative_weight2 = numpy.dot(derivative_z2.T, activation1).T
    derivative_bias2 = numpy.sum(derivative_z2, axis=0)

    derivative_z1 = numpy.dot(derivative_z2, weight2.T) * derivative_sigmoid(activation1)
    derivative_weight1 = numpy.dot(derivative_z1.T, activation0).T
    derivative_bias1 = numpy.sum(derivative_z1, axis=0)

def update_weight(learning_rate):
    global weight3 
    global weight2 
    global weight1

    global bias3
    global bias2
    global bias1

    weight3 = weight3 - (learning_rate * derivative_weight3)
    weight2 = weight2 - (learning_rate * derivative_weight2)
    weight1 = weight1 - (learning_rate * derivative_weight1)

    bias3 = bias3 - (learning_rate * derivative_bias3)
    bias2 = bias2 - (learning_rate * derivative_bias2)
    bias1 = bias1 - (learning_rate * derivative_bias1)

def train_model(train_images, train_labels):
    for epoch in range(EPOCHS):

        loss_for_epoch = 0
        for batch_number in range(len(train_images) // BATCH_SIZE):
            start_index = batch_number * BATCH_SIZE
            end_index = (batch_number + 1) * BATCH_SIZE
            train_image_for_this_batch = train_images[start_index: end_index]
            train_label_for_this_batch = train_labels[start_index: end_index]
            forward_pass(train_image_for_this_batch)
            loss = loss_function(train_label_for_this_batch)
            loss_for_epoch += loss
            backward_propagation(train_label_for_this_batch)
            update_weight(LEARNING_RATE)
        print(f"epoch: {epoch} | sum loss: {loss_for_epoch} | avg loss: {loss_for_epoch / BATCH_SIZE}")
        
        now = getTimeNowInMilli()
        print(f"NOW: {now}")
        print(f"Now - Start_time: {now - START_TIME} | using: {(now - START_TIME) / (1000 * 60)} min")
        if now - START_TIME >= MAX_TIME_FOR_TRAINING:
            print(f"it is in if break")
            break
        shuffle_train_data()

def test_model(test_images):
    output = forward_pass(test_images)
    output = numpy.argmax(output, 1)
    write_to_file(TEST_OUTPUT_PATH, output)

def shuffle_train_data(): 
    global train_images
    global train_labels

    shuffled_index = [i for i in range(train_images.shape[0])]
    numpy.random.shuffle(shuffled_index)
    train_images = train_images[shuffled_index]
    train_labels = train_labels[shuffled_index]

train_model(train_images, train_labels_new_format)

test_images = read_file(TEST_IMAGE_PATH)
test_model(test_images)
END_TIME = getTimeNowInMilli()
print(f"terminate_program_at: {END_TIME}")
print(f"using: {END_TIME - START_TIME} millisec | {(END_TIME - START_TIME) / 1000} sec | {(END_TIME - START_TIME) / (1000 * 60)} mim")