#include<bits/stdc++.h>
#include "./Eigen3/Dense"
#include "grad.h"
#include <random>
#define Vec vector
#define Pair pair
#define RNG std::default_random_engine
#define UniqPtr unique_ptr
using namespace std;

/// The meta-data of composite feature
struct CpFeatureMeta {
  /// The value cap of each categorical attribute
  Vec<int> enum_cap;
  /// The total number of numerical attribute
  int num_size;
};

struct CpFeature {
  Vec<int> enum_;
  Vec<double> num_;
};


class ForwordBaseNetwork {
public:
        const int enum_size_, num_size_;
        const Vec<int> enum_cap_;
        const Vec<int> idx_offset_;
        const int dim_;

        const Vec<int> layer_neuron_num_;
        const int layer_num_;
        Vec<Eigen::VectorXd> layer_;
        Vec<Eigen::MatrixXd> weights_;
        Vec<Eigen::VectorXd> bias_;
        Vec<Eigen::VectorXd> delta_error_;
        double init_scale;
        string activation_function;
        double learning_rate;
        const double reg_coeff = 0.1;
        int mini_batch_size;

    ForwordBaseNetwork(const CpFeatureMeta& meta,  Vec<int> idx_offset, Vec<int> layer_neuron_num, double scale) : 
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
        bias_.resize(layer_num_ - 1);
        for(int i = 0; i < layer_num_ - 1; i++) {
            weights_[i] = Eigen::MatrixXd(layer_[i+1].rows(), layer_[i].rows());
            bias_[i] = Eigen::VectorXd(layer_[i+1].rows());
        }
    }

    void Reset(RNG& gen){
        std::uniform_real_distribution<double> dist(-1, 1);
        //initialize weights matrix
        for(int i = 0; i  < weights_.size(); i++)
            for(int j = 0; j < weights_[i].rows(); j++)
                for(int k = 0; k < weights_[i].cols(); k++)
                    weights_[i](j, k) = dist(gen) * init_scale;
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
            weights_[i] =  weights_[i] * (1 - reg_coeff * learning_rate * 2)- learning_rate * (delta_error_[i] * layer_[i].transpose());
            bias_[i] = bias_[i] - learning_rate * delta_error_[i];
        }
    }

    void FitMiniBatch(const Vec<Pair<CpFeature, Grad>>& grad) {
        double output_error = 0;
        for(int i = 0; i < grad.size(); i++)
            output_error = grad[i].second.left;
        backward(output_error / grad.size());        
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
        return max(0.0, x);
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

    void activeFunction(Eigen::VectorXd & ans, Eigen::MatrixXd && x, string type) {
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

    void derivativeFunction(Eigen::VectorXd & ans, Eigen::VectorXd & x, string type) {
        if(type == "none")
            for(int i = 0; i < x.rows(); i++)
                ans(i, 0) = 1;
        else if(type == "sigmod")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = sigmod(x(i, 0)) * (1 - sigmod(x(i, 0)));
        else if(type == "tanh")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = 1 - tanh(x(i, 0)) * tanh(x(i, 0));
        else if(type == "Relu")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = (x(i, 0) > 0) ? 1 : 0;
        else if(type == "leaky_ReLU")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = (x(i, 0) > 0) ? 1 : 0.25;
        else if(type == "ELU")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = (x(i, 0) > 0) ? 1 : 0.25 * exp(x(i, 0));
    }

    static UniqPtr<ForwordBaseNetwork> Construct(const CpFeatureMeta& meta, double scale) noexcept {
        Vec<int> idx_offset(1, 0);
        int offset = 0;
        for (int cap : meta.enum_cap) {
            offset += cap;
            idx_offset.push_back(offset);
        }

        Vec<int> layer_neuron_num{idx_offset.back() + meta.num_size, 2, 3, 1};
        ForwordBaseNetwork model(meta, idx_offset, layer_neuron_num, scale);
        return std::make_unique<ForwordBaseNetwork>(std::move(model));
    }
    
};


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
        Vec<Eigen::VectorXd> delta_error_;
        double init_scale;
        string activation_function;
        double learning_rate;
        const double reg_coeff = 0.1;
        const double loss_rate = 0.9;

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
            momentum_[i] = loss_rate * momentum_[i] + learning_rate * (delta_error_[i] * layer_[i].transpose());
            weights_[i] =  weights_[i] * (1 - reg_coeff * learning_rate * 2) - momentum_[i]; 
            bias_[i] = bias_[i] - learning_rate * delta_error_[i];
        }
    }

    void FitMiniBatch(const Vec<Pair<CpFeature, Grad>>& grad) {
        double output_error = 0;
        for(int i = 0; i < grad.size(); i++)
            output_error = grad[i].second.left;
        backward(output_error / grad.size());        
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
        return max(0.0, x);
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

    void activeFunction(Eigen::VectorXd & ans, Eigen::MatrixXd && x, string type) {
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

    void derivativeFunction(Eigen::VectorXd & ans, Eigen::VectorXd & x, string type) {
        if(type == "none")
            for(int i = 0; i < x.rows(); i++)
                ans(i, 0) = 1;
        else if(type == "sigmod")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = sigmod(x(i, 0)) * (1 - sigmod(x(i, 0)));
        else if(type == "tanh")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = 1 - tanh(x(i, 0)) * tanh(x(i, 0));
        else if(type == "Relu")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = (x(i, 0) > 0) ? 1 : 0;
        else if(type == "leaky_ReLU")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = (x(i, 0) > 0) ? 1 : 0.25;
        else if(type == "ELU")
            for(int i = 0; i < x.rows(); i++)
                    ans(i, 0) = (x(i, 0) > 0) ? 1 : 0.25 * exp(x(i, 0));
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


const char *s1[6] = {"base_data_none.txt", "base_data_sigmod.txt", 
"base_data_tanh.txt", "base_data_ReLU.txt", "base_data_leaky_ReLU.txt", "base_data_ELU.txt"};

const char *s2[6] = {"momentum_data_none.txt", "momentum_data_sigmod.txt", 
"momentum_data_tanh.txt", "momentum_data_ReLU.txt", "momentum_data_leaky_ReLU.txt", "momentum_data_ELU.txt"};

void solve(int type, string active)
{ 
    CpFeatureMeta meta{ Vec<int>{2}, 2 };
    auto model = ForwordBaseNetwork::Construct(meta, 1);
    model->activation_function = active;
    
    cout << "ForwordBaseNetwork:  " << active << endl;

    std::default_random_engine gen(910109);
    std::uniform_int_distribution<int> dist(0, 1);
    double realScale = 1;
    std::uniform_real_distribution<double> dist2(-10, 10);
    std::normal_distribution<double> noise(0, 1);
    model->Reset(gen);
    FILE *fin = fopen(s1[type], "w");
    //产生测试集

  for (int epoch = 0; epoch < 5000; epoch++)
  {
      for (int i = 0; i < 200; ++i)
      {
          Vec<Pair<CpFeature, Grad>> batch;
          for (int j = 0; j < 20; ++j) {
            int a = dist(gen), b = dist(gen), c = dist(gen);
            double d = dist2(gen), e = dist2(gen), f = dist2(gen); //+ noise(gen);
            double score = (a == 0) ? d : e;
            CpFeature feature{ Vec<int>{a}, Vec<double>{d, e} };
            for(auto &x : feature.num_)
              x /= realScale;
            //printf("%.6f", feature.num_[0]);
            double val = model->EvalScore(feature);
            double grad = 2 * (val - score);
            batch.push_back(std::make_pair(feature, Grad{grad, grad}));
          }
          model->FitMiniBatch(batch);
       }
      double mse = 0;
      for (int i = 0; i < 1000; ++i) {
            int a = dist(gen), b = dist(gen), c = dist(gen);
            double d = dist2(gen), e = dist2(gen), f = dist2(gen); //+ noise(gen);
            double score = (a == 0) ? d : e;
            CpFeature feature{ Vec<int>{a}, Vec<double>{d, e} };
        for(auto &x : feature.num_)
               x /= realScale;
        double val = model->EvalScore(feature);
        //fprintf(fin, "\n a,b,c: %d %d %d,   d, e, f: %.8f %.8f %.8f,   score: %8f,   val:%.8f\n", a, b, c, d, e, f, score, val);
        mse += (val - score) * (val - score);
    }
    fprintf(fin, "%.8f\n", mse / 1000);
  }

  
  //REQUIRE(mse / 100000 < 0.1);
  fclose(fin);
  double mse = 0;
  for (int i = 0; i < 100000; ++i) {
     int a = dist(gen), b = dist(gen), c = dist(gen);
    double d = dist2(gen), e = dist2(gen), f = dist2(gen); //+ noise(gen);
    double score = (a == 0) ? d : e;
    CpFeature feature{ Vec<int>{a}, Vec<double>{d, e} };
  
     for(auto &x : feature.num_)
              x /= realScale;
    double val = model->EvalScore(feature);
    mse += (val - score) * (val - score);
  }
  printf("mse: %.8f\n\n", mse /  100000);
}


int main()
{
   string active[6] = {"none", "sigmod", "tanh", "ReLU", "leaky_ReLU", "ELU"};
   for(int i = 0; i < 6; i++)
        solve(i, active[i]);
}