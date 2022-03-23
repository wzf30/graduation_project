#include "ForwordBaseNetwork.h"
#include "ForwordMomentumNetwork.h"
using namespace std;



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

  for (int epoch = 0; epoch < 50000; epoch++)
  {
      for (int i = 0; i < 200; ++i)
      {
          Vec<Pair<CpFeature, Grad>> batch;
          for (int j = 0; j < 1; ++j) {
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