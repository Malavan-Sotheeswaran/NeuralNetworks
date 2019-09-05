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

class NN::InputLayer : NN::Layer {
public:
    InputLayer(int size) : Layer(0, size, NN::Layer::LayerType::Input), nodes(size) {}

    void load_input(std::vector<double> input) {
        for(int i = 0; i < nodes.size(); i++) {
            nodes[i].set_value(input[i]);
        }
    }

    void feed_layer() {}

    void forward() {}
    void backprop() {}

    std::vector<Node>::iterator begin() { return nodes.begin(); }
    std::vector<Node>::iterator end() { return nodes.end(); }
private:
    std::vector<Node> nodes;
};

class NN::DenseLayer : NN::Layer {
public:
    DenseLayer(int in_size, int out_size) : Layer(in_size, out_size, NN::Layer::LayerType::Dense), nodes(out_size) {}

    void feed_layer(Layer& layer) {
        for(auto in_node : layer) {
            for(auto node : nodes) {
                node.add_input(in_node);
            }
        }
    }

    void forward() {
        for(auto node : nodes){
            node.forward();
        }
    }

    void backprop() {
        for(auto node : nodes){
            node.backprop();
        }
    }

    std::vector<Node>::iterator begin() { return nodes.begin(); }
    std::vector<Node>::iterator end() { return nodes.end(); }
private:
    std::vector<NN::Node> nodes;
};

void NN::NeuralNetwork::add_layer(Layer layer){
    if(network.empty()) {
        layer.feed_layer(input);
    }
    else {
        layer.feed_layer(network.back());
    }
    network.push_back(layer);
    output.clear();
    for(auto node : network.back()) {
        output.push_back(Connection(node));
    }
}

std::vector<double> NN::NeuralNetwork::operator()(std::vector<double> input) {
    this->input.load_input(input);
    for(auto layer : network) {
        layer.forward();
    }
    std::vector<double> out(output.size());
    for(int i = 0; i < output.size(); i++) {
        out[i] = output[i].get();
    }
    return out;
}

void NN::NeuralNetwork::train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> targets) {
    double loss = 0;
    for(int i = 0; i < inputs.size(); i++) {
        loss += 0; //loss_function((*this)(inputs[i]), targets[i]);
    }
    std::vector<double> output_gradient(output.size());// = loss_function.grad(loss);
    for(int i = 0; i < output.size(); i++) {
        output[i].backprop(output_gradient[i]);
    }
    for(auto layer : util::reverse<std::vector<Layer>>(network)) {
        layer.backprop();
    }
}

int main() {
    auto NN = NN::NeuralNetwork();
    NN.add_layer(NN::DenseLayer(5,10));
    NN.add_layer(NN::DenseLayer(10,5));
    NN.add_layer(NN::DenseLayer(5,1));
}