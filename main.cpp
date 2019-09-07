#include "NN.hpp"
int main() {
    auto NN = NN::NeuralNetwork(5);
    NN::DenseLayer input(5,10);
    NN::DenseLayer hidden(10,5);
    NN::DenseLayer output(5,1);
    NN.add_layer(input);
    NN.add_layer(hidden);
    NN.add_layer(output);
}