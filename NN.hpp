#pragma once
#include <vector>
namespace util {
    template <class T> class reverse{
    public:
        reverse(T obj) { obj = obj; }
        auto begin() { return obj.rbegin(); }
        auto end() { return obj.rend(); }
    private:
        T obj;
    };
}

namespace NN {
    class Connection;

    double default_loss(std::vector<double>& output, std::vector<double>& target);
    std::vector<double> default_grad(std::vector<double>& output, std::vector<double>& target);

    class LossFunction{
    public:
        LossFunction() : function{default_loss}, gradient{default_grad} {}
        LossFunction(double (*f) (std::vector<double>&, std::vector<double>&), std::vector<double> (*g) (std::vector<double>&, std::vector<double>&)) : function{f}, gradient{g} {}
        double operator()(std::vector<double>& output, std::vector<double>& target) { return function(output, target); }
        std::vector<double> grad(std::vector<double>& output, std::vector<double>& target) { return gradient(output, target); }
    private:
        double (*function) (std::vector<double>&, std::vector<double>&);
        std::vector<double> (*gradient)(std::vector<double>&, std::vector<double>&);
    };

    class Node {
    public: 
        Node() {}
        void forward();
        double get_value() { return value; }
        void set_value(double value) { value = value; }
        void backprop();
        void add_input(Node& node);
        void update_gradient(double child_gradient) { gradient += child_gradient; }
    private:
        std::vector<Connection> inputs;
        double value = 0;
        double gradient = 0;
    };

    class Connection {
    public:
        Connection(Node& node) : node{node} {}
        double get() { return node.get_value()*weight; }
        double get_output() { return node.get_value(); }
        void backprop(double gradient);
    private:
        Node& node;
        double weight = 0;
    };

    class Layer {
    public:
        enum class LayerType {
            Input,
            Dense,
            Convolutional,
            Pooling,
            Activation
        };
        Layer(size_t in_size, size_t out_size, LayerType type) : in_size{in_size}, out_size{out_size}, type{type} {}
        virtual void forward() = 0;
        virtual void backprop() = 0;
        virtual std::vector<Node>::iterator begin() = 0;
        virtual std::vector<Node>::iterator end() = 0;
        virtual Node& operator[](size_t index) = 0;
        virtual void feed_layer(Layer& layer) = 0;
        size_t size() { return out_size; }
    private:
        size_t in_size;
        size_t out_size;
        LayerType type;
    };

    class InputLayer : public Layer {
    public:
        InputLayer(size_t size) : Layer(0, size, NN::Layer::LayerType::Input), nodes(size) {}
        void load_input(std::vector<double> input);
        void feed_layer(Layer& layer) {}
        void forward() {}
        void backprop() {}
        Node& operator[](size_t index) { return nodes[index]; }
        std::vector<Node>::iterator begin() { return nodes.begin(); }
        std::vector<Node>::iterator end() { return nodes.end(); }
    private:
        std::vector<Node> nodes;
    };

    class DenseLayer : public Layer {
    public:
        DenseLayer(size_t in_size, size_t out_size) : Layer(in_size, out_size, NN::Layer::LayerType::Dense), nodes(out_size) {}
        void feed_layer(Layer& layer);
        void forward();
        void backprop();
        Node& operator[](size_t index) { return nodes[index]; }
        std::vector<Node>::iterator begin() { return nodes.begin(); }
        std::vector<Node>::iterator end() { return nodes.end(); }
    private:
        std::vector<Node> nodes;
    };

    class NeuralNetwork {
    public:
        NeuralNetwork(size_t input_size) : input(input_size) {}
        void add_layer(Layer& layer);
        std::vector<double> operator()(std::vector<double> input);
        void train(std::vector<std::vector<double>> input, std::vector<std::vector<double>> target);
    private:
        InputLayer input;
        std::vector<Layer*> network;
        LossFunction loss_function;
        friend class NNBuilder;
    };
}