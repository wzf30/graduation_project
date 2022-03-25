#pragma once

#include "grad.h"

class ForwordMomentumNetwork {
public:
        const int enum_size_, num_size_;
        const Vec<int> enum_cap_;
        const Vec<int> idx_offset_;
        const int dim_;

        const Vec<int> layer_neuron_num_;
        const int layer_num_;
        Vec<Eigen::VectorXd> layer_;
        Vec<Eigen::MatrixXd> weights_;
        Vec<Eigen::MatrixXd> momentum_;
        Vec<Eigen::VectorXd> bias_;
        Vec<Eigen::MatrixXd> momentum_tmp;
        Vec<Eigen::VectorXd> bias_tmp;
        Vec<Eigen::VectorXd> delta_error_;
        double init_scale;
        std::string activation_function;
        double learning_rate;
        const double reg_coeff = 0.1;
        const double loss_rate = 0.9;
        int mini_batch_size;

    ForwordMomentumNetwork(const CpFeatureMeta& meta,  Vec<int> idx_offset, Vec<int> layer_neuron_num,  double scale) : 
      enum_size_(static_cast<int>(meta.enum_cap.size())),
      num_size_(meta.num_size),
      enum_cap_(meta.enum_cap),
      idx_offset_(idx_offset),
      dim_(idx_offset.back() + meta.num_size),
      layer_neuron_num_(layer_neuron_num),
      layer_num_(layer_neuron_num.size()),
      init_scale(scale),
      activation_function("none") {
        learning_rate =  0.0001;
        // generate the layer(store the output of every layer)
        layer_.resize(layer_num_);
        for(int i = 0; i < layer_num_; i++)
            layer_[i] = Eigen::VectorXd(layer_neuron_num_[i]);
        
        //generate weight matrix and bias vector
        weights_.resize(layer_num_ - 1);
        momentum_.resize(layer_num_ - 1);
        bias_.resize(layer_num_ - 1);
        for(int i = 0; i < layer_num_ - 1; i++) {
            weights_[i] = Eigen::MatrixXd(layer_[i+1].rows(), layer_[i].rows());
            momentum_[i] = Eigen::MatrixXd(layer_[i+1].rows(), layer_[i].rows());
            bias_[i] = Eigen::VectorXd(layer_[i+1].rows());
        }

        //generate tmp_momentum matrix and bias vector
        momentum_tmp.resize(layer_num_ - 1);
        bias_tmp.resize(layer_num_ - 1);
        for(int i = 0; i < layer_num_ - 1; i++) {
            momentum_tmp[i] = Eigen::MatrixXd(layer_[i+1].rows(), layer_[i].rows());
            bias_tmp[i] = Eigen::VectorXd(layer_[i+1].rows());
        }
    }

    void Reset(RNG& gen){
        std::uniform_real_distribution<double> dist(-1, 1);
        //initialize weights and momentum matrix
        for(int i = 0; i  < weights_.size(); i++)
            for(int j = 0; j < weights_[i].rows(); j++)
                for(int k = 0; k < weights_[i].cols(); k++)
                    weights_[i](j, k) = dist(gen) * init_scale,
                    momentum_[i](j, k) = 0;
        //initialize bias vector
        for(int i = 0; i < bias_.size(); i++)
            for(int j = 0; j < bias_[i].rows(); j++)
                bias_[i](j ,0) = dist(gen) * init_scale;
    }

    void forward() {
        for(int i = 0; i < layer_num_ - 2; i++)
            activeFunction(layer_[i+1], weights_[i] * layer_[i] + bias_[i], activation_function);
        // the output layer don't need activeFunction
        layer_[layer_num_ - 1] = weights_[layer_num_-2] * layer_[layer_num_-2] + bias_[layer_num_-2];
    }

    void backward(double output_error) {
        delta_error_.resize(layer_num_ - 1);
        for (int i = delta_error_.size() - 1; i >= 0; i--) {
            Eigen::VectorXd dx = Eigen::VectorXd(layer_neuron_num_[i+1]);
            
            if (i == delta_error_.size() - 1)
                dx(0, 0) = 1,
                delta_error_[i] = output_error * dx;
            else  //Hidden layer delta error
                derivativeFunction(dx, layer_[i + 1], activation_function),
                delta_error_[i] = dx.cwiseProduct(weights_[i+1].transpose() * delta_error_[i + 1]);
        }

        for(int i = 0; i < weights_.size(); i++) {
            momentum_tmp[i] +=  learning_rate * (delta_error_[i] * layer_[i].transpose());
            bias_tmp[i] += learning_rate * delta_error_[i];
            // momentum_[i] = loss_rate * momentum_[i] + learning_rate * (delta_error_[i] * layer_[i].transpose());
            // weights_[i] =  weights_[i] * (1 - reg_coeff * learning_rate * 2) - momentum_[i]; 
            // bias_[i] = bias_[i] - learning_rate * delta_error_[i];
        }
    }

    void FitMiniBatch(const Vec<Pair<CpFeature, Grad>>& grad) {
        // double output_error = 0;
        // for(int i = 0; i < grad.size(); i++)
        //     output_error = grad[i].second.left;
        // backward(output_error / grad.size());  

        //initialize tmp_momentum matrix
        for(int i = 0; i  < weights_.size(); i++)
            for(int j = 0; j < weights_[i].rows(); j++)
                for(int k = 0; k < weights_[i].cols(); k++)
                    momentum_tmp[i](j, k) = 0;

        //initialize tmp_bias vector
        for(int i = 0; i < bias_.size(); i++)
            for(int j = 0; j < bias_[i].rows(); j++)
                bias_tmp[i](j ,0) = 0;

        int batch_num = grad.size();
        for(int i = 0; i < batch_num; i++) {
            EvalScore(grad[i].first);
            backward(grad[i].second.left);
        }

        for(int i = 0; i  < weights_.size(); i++) {
            momentum_[i] = loss_rate * momentum_[i] + (1.0 / batch_num) * momentum_tmp[i]; 
            weights_[i] = weights_[i] * (1 - reg_coeff * learning_rate * 2) - momentum_[i];
            bias_[i] = bias_[i] - (1.0 / batch_num) * bias_tmp[i];
        }
    } 

    double EvalScore(const CpFeature& feature)   {
        for(int i = 0; i < enum_size_; i++) {
            for(int j = idx_offset_[i]; j < idx_offset_[i+1]; j++)
                layer_[0](j, 0) = 0;
            layer_[0](idx_offset_[i]+feature.enum_[i], 0) = 1;
        }
        for(int i = 0; i < num_size_; i++)
            layer_[0](idx_offset_.back() + i, 0) = feature.num_[i];
        forward();
        return layer_[layer_num_ - 1](0, 0);
    }

    double sigmod(double x) {
        return 1 / (1 + exp(-x));
    }

    double Relu(double x) {
        return std::max(0.0, x);
    }

    double tanh(double x) {
        return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
    }

    double leaky_Relu(double x) {
        return (x > 0) ? x : 0.25 * x;
    }

    double ELU(double x) {
        return (x > 0) ? x : 0.25 * (exp(x) - 1);
    }

    void activeFunction(Eigen::VectorXd & ans, Eigen::MatrixXd && x, std::string type) {
        if(type == "none")
            for(int i = 0; i < x.rows(); i++)
                ans(i, 0) = x(i, 0);
        else if(type == "sigmod")
            for(int i = 0; i < x.rows(); i++)         
                    ans(i, 0) = sigmod(x(i, 0));
        else if(type == "tanh")
            for(int i = 0; i < x.rows(); i++)         
                    ans(i, 0) = tanh(x(i, 0));
        else if(type == "ReLU")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = Relu(x(i, 0));
        else if(type == "leaky_ReLU")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = leaky_Relu(x(i, 0));
        else if(type == "ELU")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = ELU(x(i, 0));
    }

    void derivativeFunction(Eigen::VectorXd & ans, Eigen::VectorXd & x, std::string type) {
        if(type == "none")
            for(int i = 0; i < x.rows(); i++)
                ans(i, 0) = 1;
        else if(type == "sigmod")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = x(i, 0) * (1 - x(i, 0));
        else if(type == "tanh")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = 1 - x(i, 0) * x(i, 0);
        else if(type == "Relu")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = (x(i, 0) > 0) ? 1 : 0;
        else if(type == "leaky_ReLU")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = (x(i, 0) > 0) ? 1 : 0.25;
        else if(type == "ELU")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = (x(i, 0) > 0) ? 1 : x(i, 0) + 0.25;
    }

    static UniqPtr<ForwordMomentumNetwork> Construct(const CpFeatureMeta& meta, double scale) noexcept {
        Vec<int> idx_offset(1, 0);
        int offset = 0;
        for (int cap : meta.enum_cap) {
            offset += cap;
            idx_offset.push_back(offset);
        }

        Vec<int> layer_neuron_num{idx_offset.back() + meta.num_size, 3, 2, 1};
        ForwordMomentumNetwork model(meta, idx_offset, layer_neuron_num, scale);
        return std::make_unique<ForwordMomentumNetwork>(std::move(model));
    }
    
};
