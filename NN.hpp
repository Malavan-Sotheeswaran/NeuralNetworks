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

    // double default_loss(std::vector<double> input, std::vector<double> target) { return 0; }

    // class LossFunction{
    // public:
    //     LossFunction() : function{default_loss} {}
    //     LossFunction(double (*f) (std::vector<double>, std::vector<double>)) : function{f} {}
    //     double operator()(std::vector<double> input, std::vector<double> target) { return function(input, target); }
    //     std::vector<double> grad(double) { return std::vector<double>(); }
    // private:
    //     double (*function) (std::vector<double>,std::vector<double>);
    // };

    class Node {
    public: 
        Node() {}
        void forward();
        double get_value() { return value; }
        void set_value(double value) { value = value; }
        void backprop();
        void add_input(Node& node);
    private:
        void update_gradient(double child_gradient) { gradient += child_gradient; }

        std::vector<Connection> inputs;
        double value = 0;
        double gradient = 0;
        friend class Connection;
    };

    class Connection {
    public:
        Connection(Node& node) : node{node} {}
        double get() { return node.get_value()*weight; }
        double get_output() { return node.get_value(); }
        double backprop(double gradient) {
            node.update_gradient(gradient*weight);
            weight -= node.get_value()*gradient;
        }
    private:
        Node& node;
        double weight = 0;
    };

    class Layer {
    public:
        Layer(int in_size, int out_size, LayerType type) : in_size{in_size}, out_size{out_size}, type{type} {}
        enum class LayerType {
            Input,
            Dense,
            Convolutional,
            Pooling,
            Activation
        };
        virtual void forward() = 0;
        virtual void backprop() = 0;
        virtual std::vector<Node>::iterator begin() = 0;
        virtual std::vector<Node>::iterator end() = 0;

        virtual void feed_layer(Layer& layer) = 0;
    private:
        int in_size;
        int out_size;
        LayerType type;
    };

    class InputLayer;
    class DenseLayer;
    // class ConvolutionalLayer;
    // class PoolingLayer;
    // class ActivationLayer;

    class NeuralNetwork {
    public:
        NeuralNetwork() {}
        void add_layer(Layer layer);
        std::vector<double> operator()(std::vector<double> input);
        void train(std::vector<std::vector<double>> input, std::vector<std::vector<double>> target);
    private:
        InputLayer input;
        std::vector<Layer> network;
        std::vector<Connection> output;
        //LossFunction loss_function;
        friend class NNBuilder;
    };
}