# Code reference: https://www.udemy.com/course/machine-learning-build-a-neural-network-in-77-lines-of-code/
import math
import random
class NeuralNetwork():
    def __init__(self):
        #random.seed(1)
        self.weights = [random.uniform(-1, 1) for _ in range(3)]

    # Make a prediction with the neural network
    def think(self,neuron_input):
        print("think==", neuron_input)
        sum_of_weighted_inputs = self.__sum_of_weighted_inputs(neuron_input)
        print("sum_of_weighted_inputs==", sum_of_weighted_inputs)
        neuron_output = self.__sigmoid(sum_of_weighted_inputs)
        print("neuron_output==", neuron_output)
        return neuron_output

    # Adjust the weights of the neural network to minimise the error for the training set
    def train(self,training_set_examples,number_of_iterations):
        for iteration in range(number_of_iterations):
            for training_set_example in training_set_examples:
                print("training_set_example=", training_set_examples[0])
                predicted_output = self.think(training_set_example["inputs"])
                actual_output = training_set_example["outputs"]
                error_in_output = actual_output - predicted_output
                for index in range(len(self.weights)):
                    neuron_input = training_set_example["inputs"][index]
                    adjustment = neuron_input * error_in_output * self.__sigmoid_gradient(predicted_output)
                    print("adjustment==", adjustment)
                    print("self.weights[index]==", self.weights[index])
                    self.weights[index] += adjustment
                    print("after adjustment self.weights[index]==", self.weights[index])

    # Activation function
    def __sigmoid(self, sum_of_weighted_inputs):
        return 1 / (1 + math.exp(-sum_of_weighted_inputs))

    # loss function for the output layer
    def __sigmoid_gradient(self,neuron_output):
        return neuron_output * (1 - neuron_output)

    # Calculate the neuron's weighted sum
    def __sum_of_weighted_inputs(self,neuron_inputs):
        sum_of_weighted_inputs = 0
        for index, neuron_input in enumerate(neuron_inputs):
            sum_of_weighted_inputs += self.weights[index] * neuron_input
        return sum_of_weighted_inputs


neural_network = NeuralNetwork()
print("Random starting weights: " + str(neural_network.weights))

training_set_examples = [
        {"inputs": [0, 0, 1], "outputs": 0},
        {"inputs": [1, 1, 1], "outputs": 1},
        {"inputs": [1, 0, 1], "outputs": 1},
        {"inputs": [0, 1, 1], "outputs": 0},
    ]

# Train the nn using 10000 iterations 
neural_network.train(training_set_examples, number_of_iterations=1)
print("New weights after training: " + str(neural_network.weights))
# Make a prediction for a new situation
new_situation = [0, 1, 1]
prediction = neural_network.think(new_situation)
print("Prediction for the new situation [1,0,0] -> ? " + str(prediction))