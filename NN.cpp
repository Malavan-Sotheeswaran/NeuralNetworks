#include "NN.hpp"
double NN::DefaultLoss::operator()(std::vector<double>& output, std::vector<double>& target) { 
    double result = 0;
    for(size_t i; i < output.size(); i++) {
        result += 0.5*(output[i] - target[i])*(output[i] - target[i]);
    }
    return result;
}

std::vector<double> NN::DefaultLoss::grad(std::vector<double>& output, std::vector<double>& target) {
    std::vector<double> result(output.size());
    for(size_t i; i < output.size(); i++) {
        result[i] = (output[i] - target[i]);
    }
    return result;
}

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

void NN::Connection::backprop(double gradient) {
    node.update_gradient(gradient*weight);
    weight -= node.get_value()*gradient;
}

void NN::InputLayer::load_input(std::vector<double> input) {
    for(size_t i = 0; i < nodes.size(); i++) {
        nodes[i].set_value(input[i]);
    }
}

void NN::DenseLayer::feed_layer(Layer& layer) {
    for(auto in_node : layer) {
        for(auto node : nodes) {
            node.add_input(in_node);
        }
    }
}
void NN::DenseLayer::forward() {
    for(auto node : nodes){
        node.forward();
    }
}
void NN::DenseLayer::backprop() {
    for(auto node : nodes){
        node.backprop();
    }
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
        std::vector<double> output_gradient = loss_function->grad(output, targets[i]);
        for(size_t i = 0; i < (*network.back()).size(); i++) {
            (*network.back())[i].update_gradient(output_gradient[i]);
        }
        for(auto layer : util::reverse<std::vector<Layer*>>(network)) {
            layer->backprop();
        }
    }
}
