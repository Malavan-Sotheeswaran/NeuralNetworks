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
}

std::vector<double> NN::NeuralNetwork::operator()(std::vector<double> input) {
    this->input.load_input(input);
    for(auto layer : network) {
        layer->forward();
    }
    std::vector<double> out((*network.back()).size());
    for(size_t i = 0; i < (*network.back()).size(); i++) {
        out[i] = (*network.back())[i].get_value();
    }
    return out;
}

void NN::NeuralNetwork::train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets) {
    for(size_t i = 0; i < inputs.size(); i++) {
        std::vector<double> output = (*this)(inputs[i]);
        std::vector<double> output_gradient = loss_function.grad(output, targets[i]);
        for(size_t i = 0; i < (*network.back()).size(); i++) {
            (*network.back())[i].update_gradient(output_gradient[i]);
        }
        for(auto layer : util::reverse<std::vector<Layer*>>(network)) {
            layer->backprop();
        }
    }
}
