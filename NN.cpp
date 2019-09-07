#include "NN.hpp"
void NN::Node::forward() {
    value = 0;
    for(auto connection : inputs) {
        value += connection.get();
    }
}

void NN::Node::backprop() {
    for(auto connection : inputs) {
        connection.backprop(gradient);
    }
    gradient = 0;
}

void NN::Node::add_input(Node& node) { 
    inputs.push_back(Connection(node)); 
}

void NN::NeuralNetwork::add_layer(Layer& layer){
    if(network.empty()) {
        layer.feed_layer(input);
    }
    else {
        layer.feed_layer(*network.back());
    }
    network.push_back(&layer);
    output.clear();
    for(auto node : *network.back()) {
        output.push_back(Connection(node));
    }
}

std::vector<double> NN::NeuralNetwork::operator()(std::vector<double> input) {
    this->input.load_input(input);
    for(auto layer : network) {
        layer->forward();
    }
    std::vector<double> out(output.size());
    for(size_t i = 0; i < output.size(); i++) {
        out[i] = output[i].get();
    }
    return out;
}

void NN::NeuralNetwork::train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets) {
    double loss = 0;
    for(size_t i = 0; i < inputs.size(); i++) {
        loss += 0; //loss_function((*this)(inputs[i]), targets[i]);
    }
    std::vector<double> output_gradient(output.size());// = loss_function.grad(loss);
    for(size_t i = 0; i < output.size(); i++) {
        output[i].backprop(output_gradient[i]);
    }
    for(auto layer : util::reverse<std::vector<Layer*>>(network)) {
        layer->backprop();
    }
}
