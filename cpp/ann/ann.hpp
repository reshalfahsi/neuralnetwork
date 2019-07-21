#ifndef ANN_HPP
#define ANN_HPP

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <vector>
#include <random>
#include <string>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <iosfwd>
#include <fstream>
#include <istream>
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>
#include <sstream>
#include <string>

#define EPSILON 3e-3
#define INF 9999999
#define RELU_STEP 2e-2
#define BIAS M_PI*31e-4
#define BETA 0.90
#define EPS 1e-8
#define BETA_TWO 0.999
#define ACCURACY_TOLEARANCE 31e-4

#define ADAM_OPTIMIZER

/* STILL UNSTABLE, NEED TO BE DEVELOPED */

/* LOOKING FOR REFERENCE?
 *
 * https://gluon.mxnet.io/chapter01_crashcourse/preface.html
 *
 */

typedef std::vector<std::vector<double>> Data;
typedef std::pair<Data,Data> BatchData;

namespace fukuro {

struct Layer{

    int node;

    /* BATCH NORMIES */
    double gamma;
    double dgamma;
    double beta;
    double dbeta;
    bool isFirstTrain;
    double mean;
    double variance;
    Eigen::MatrixXd X_cache;
    Eigen::MatrixXd X_norm;

    /* DROPOUT */
    double drop_probability;
    Eigen::MatrixXd isAlive;

    /* AdaM Optimization */
    Eigen::MatrixXd vs;
    Eigen::MatrixXd sqrs;

    std::string type;
    Eigen::MatrixXd weight;
    Eigen::MatrixXd input;
    Eigen::MatrixXd output;
    Eigen::MatrixXd output_target;
    Eigen::MatrixXd error;
    Eigen::MatrixXd delta_weight;
    std::string activation;

};


class ANN{

public:
    ANN(int ep=500, double lr=0.01, double bs = 100, std::string path="./saved_weight/weight.dat"):
        last_accuracy(0.0), last_loss(INF)
    {

        epoch = ep;
        learning_rate = lr;
        weight_path = path;
        batch_size = bs;

    }

public:

    void add_layer(std::string type="", double node=0.0, std::string activation=""){

        Layer layer;
        layer.type = type;

        if (layer.type=="hidden")
        {
            layer.node = node;
            layer.activation = activation;

            layer.weight.resize(model[model.size()-1].node,layer.node);
            layer.delta_weight.resize(model[model.size()-1].node,layer.node);
            layer.input.resize(1,layer.node);
            layer.output.resize(1,layer.node);
            layer.output_target.resize(1,layer.node);
            layer.error.resize(1,layer.node);
            layer.vs.resize(model[model.size()-1].node,layer.node);
            layer.sqrs.resize(model[model.size()-1].node,layer.node);
        }
        else if(layer.type=="input")
        {
            layer.node = node;
            layer.activation = activation;

            layer.weight.resize(1,layer.node);
            layer.delta_weight.resize(1,layer.node);
            layer.output.resize(1,layer.node);
            layer.output_target.resize(1,layer.node);
        }
        else if(layer.type=="output")
        {
            layer.node = node;
            layer.activation = activation;

            layer.weight.resize(model[model.size()-1].node,layer.node);
            layer.delta_weight.resize(model[model.size()-1].node,layer.node);
            layer.input.resize(1,layer.node);
            layer.output.resize(1,layer.node);
            layer.output_target.resize(1,layer.node);
            layer.error.resize(1,layer.node);
            layer.vs.resize(model[model.size()-1].node,layer.node);
            layer.sqrs.resize(model[model.size()-1].node,layer.node);
        }
        else if(layer.type=="batch_normalization")
        {
            layer.node = model[model.size()-1].node;
            layer.isFirstTrain = false;

            layer.weight.resize(model[model.size()-1].node,layer.node);
            layer.output.resize(1,layer.node);
            layer.error.resize(1,layer.node);
            layer.X_cache.resize(1,layer.node);
            layer.X_norm.resize(1,layer.node);

        }
        else if(layer.type=="dropout")
        {
            layer.node = model[model.size()-1].node;

            layer.weight.resize(model[model.size()-1].node,layer.node);
            layer.output.resize(1,layer.node);
            layer.error.resize(1,layer.node);
            layer.isAlive.resize(1,layer.node);
            layer.drop_probability = node;

        }

        model.push_back(layer);

    }

    void fit(Data X_, Data Y_,std::string loss = "mse",std::string initilaizer = "lecunn_uniform"){

        auto start = std::chrono::steady_clock::now();
        std::string curr_time = "";

        if(X_.size()!=Y_.size()){
            std::cout << "Size Error!" << '\n';
            return;
        }

        //for(int idx = 1; idx<model.size(); idx++){
        //    for(int row=0; row<model[idx].delta_weight.rows(); row++){
        //        for(int col=0; col<model[idx].delta_weight.cols(); col++){
        //            model[idx].delta_weight(row,col) = std::pow(BETA*EPSILON,2)*BETA*EPSILON;
        //model[idx].delta_weight(row,col) = 0.0;
        //        }
        //    }
        //}

        struct stat info;
        if(stat(weight_path.c_str(),&info)==0)
            load_weight();
        else{
            for(int idx = 1; idx<model.size(); idx++){
                randomize_weight(model[idx].weight,initilaizer);

            }
        }

        //double accuracy = last_accuracy;
        double losses = last_loss;
        auto batchdata = std::make_pair(X_,Y_);

        for(int ep = 0; ep < epoch; ep++){

            //double accuracy_sum = 0.0;
            double predtrue = 0.0;

            //struct stat info;
            //if(stat(weight_path.c_str(),&info)==0 && std::isnan(losses))
            //{
            //    load_weight();
            //    //accuracy = last_accuracy;
            //    losses = last_loss;
            //}

            BatchData batched_data = divide_into_batch(batchdata,batch_size);
            auto X = batched_data.first;
            auto Y = batched_data.second;

            for(int data=0; data<X.size(); data++){

                for(int idx=0; idx<X[data].size(); idx++)
                    model[0].output(0,idx) = X[data][idx];


                for(int idx=0; idx<Y[data].size(); idx++)
                    model[model.size()-1].output_target(0,idx) = Y[data][idx];


                for(int l=0; l<model.size()-1; l++){
                    if(model[l+1].type=="batch_normalization"){

                        auto mean = 0.0;
                        for(int idx=0; idx<model[l].output.cols(); idx++){
                            mean += model[l].output(0,idx);
                        }
                        mean/=model[l].output.cols();
                        auto variance = 0.0;
                        for(int idx=0; idx<model[l].output.cols(); idx++){
                            variance += std::pow(model[l].output(0,idx)-mean,2);
                        }
                        variance/=model[l].output.cols();
                        auto stddev = std::sqrt(variance+EPS);
                        if(!model[l+1].isFirstTrain){
                            model[l+1].isFirstTrain = true;
                            model[l+1].gamma = std::sqrt(variance);
                            model[l+1].beta = mean;
                        }
                        for(int idx=0; idx<model[l].output.cols(); idx++){
                            model[l+1].X_cache(0,idx) = model[l].output(0,idx);
                            auto xnorm = (model[l].output(0,idx) - mean)/stddev;
                            model[l+1].X_norm(0,idx) = xnorm;
                            xnorm = model[l+1].gamma*xnorm+model[l+1].beta;
                            model[l].output(0,idx) = xnorm;
                        }
                        model[l+1].output = model[l].output;
                        model[l+1].mean = mean;
                        model[l+1].variance = variance;

                    }
                    else if(model[l+1].type=="dropout"){

                        model[l+1].output = model[l].output;
                        std::random_device rd;
                        std::mt19937 gen(rd());
                        std::bernoulli_distribution d((1.0-model[l+1].drop_probability));

                        for(int idx=0; idx<model[l+1].output.cols(); idx++){
                            int isAlive = d(gen)?1.0:0.0;
                            model[l+1].isAlive(0,idx) = isAlive;
                            model[l+1].output(0,idx) *= isAlive;
                        }

                    }
                    else{
                        model[l+1].input = model[l].output * model[l+1].weight;
                        //for(int idx=0; idx<model[l+1].input.cols(); idx++){
                        //    if(data==X.size()-1)
                        //        if(l==model.size()-2)
                        //            std::cout << "OUTPUT PER LAYER " << l+1 << ": " << "(" << 0 <<',' << idx << ") " << model[l+1].input(0,idx) << '\n';
                        //}
                        if(model[l+1].activation=="leaky_relu"){
                            model[l+1].output = activation(model[l+1].input);
                        }
                        else if(model[l+1].activation=="sigmoid"){
                            model[l+1].output = activation(model[l+1].input,"sigmoid");
                        }
                        else if(model[l+1].activation=="softmax"){
                            model[l+1].output = activation(model[l+1].input,"softmax");
                        }
                    }

                }

                /* Backprop */
                for(int b=model.size()-1; b>0; b--){

                    /* ∂Error/∂Output */
                    if(model[b].type=="batch_normalization"){

                        Eigen::MatrixXd Xmu;
                        Xmu.resize(1,model[b].node);
                        for(int idx=0; idx<model[b].X_cache.cols(); idx++){
                            Xmu(0,idx) = model[b].X_cache(0,idx) - model[b].mean;
                        }
                        auto stddev = std::sqrt(model[b].variance+EPS);
                        Eigen::MatrixXd dXnorm;
                        dXnorm.resize(1,model[b].node);
                        for(int idx=0; idx<model[b].error.cols(); idx++){
                            dXnorm(0,idx) = model[b].error(0,idx)*model[b].gamma;
                        }
                        auto dvariance = 0.0;
                        for(int idx=0; idx<model[b].error.cols(); idx++){
                            dvariance += dXnorm(0,idx)*Xmu(0,idx)*(-0.5)*std::pow(stddev,-3);
                        }
                        auto dmean = 0.0;
                        auto sum = 0.0;
                        auto manto = 0.0;
                        for(int idx=0; idx<model[b].error.cols(); idx++){
                            sum += (-1*(dXnorm(0,idx)/stddev));
                        }
                        for(int idx=0; idx<model[b].error.cols(); idx++){
                            manto += -2*(Xmu(0,idx));
                        }
                        manto/=Xmu.cols();
                        manto*=dvariance;
                        dmean = sum+manto;

                        auto dgamma = 0.0;
                        auto dbeta = 0.0;
                        for(int idx=0; idx<model[b-1].error.cols(); idx++){
                            model[b-1].error(0,idx) = (dXnorm(0,idx)/stddev) + (dvariance*2*Xmu(0,idx)) + dmean;
                            dgamma += model[b].error(0,idx)*model[b].X_norm(0,idx);
                            dbeta += model[b].error(0,idx);
                        }
                        model[b].dbeta = dbeta;
                        model[b].dgamma = dgamma;

                        model[b].beta -= learning_rate*model[b].dbeta;
                        model[b].gamma -= learning_rate*model[b].gamma;

                    }
                    else if(model[b].type=="dropout"){
                        for(int idx=0;idx<model[b].error.cols();idx++){
                            model[b-1].error(0,idx) = model[b].error(0,idx) * model[b].isAlive(0,idx);
                        }
                    }
                    else{
                        /* Loss Function */
                        //double ub_target = -INF ;
                        //double ub = -INF;
                        //double lb_target = INF;
                        //double lb = INF;
                        Eigen::MatrixXd temp_error(model[b].error.rows(),model[b].error.cols());
                        /*
                        if(loss=="categorical_cross_entropy"){
                            for(int y=0; y<model[b].output_target.cols(); y++){
                                if (ub_target < std::fabs(model[b].output_target(0,y))){
                                    ub_target = std::fabs(model[b].output_target(0,y));
                                    //std::cout << "SUM TARGET " << sum_target << '\n';
                                }
                                if (lb_target > std::fabs(model[b].output_target(0,y))){
                                    lb_target = std::fabs(model[b].output_target(0,y));
                                }
                            }
                            //std::cout << "UB TARGET " << ub_target << '\n';
                            //std::cout << "LB TARGET " << lb_target << '\n';
                            for(int y=0; y<model[b].output.cols(); y++){
                                if(ub < std::fabs(model[b].output(0,y))){
                                    ub = std::fabs(model[b].output(0,y));
                                }
                                if(lb > std::fabs(model[b].output(0,y))){
                                    lb = std::fabs(model[b].output(0,y));
                                }
                            }
                            //std::cout << "UB " << ub << '\n';
                            //std::cout << "LB " << lb << '\n';
                        }
                        */
                        for(int y=0; y<model[b].output_target.cols(); y++){
                            double error = 0.0;
                            if(loss=="mse"){
                                auto out_idx = model.size()-1;
                                //if(model[b].type=="output"){
                                if(b==out_idx){
                                    error = (model[b].output_target(0,y) - model[b].output(0,y));
                                    //std::cout << "TARGET: " << model[b].output_target(0,y) << '\n';
                                    //model[b].error(0,y) = std::pow(error,2);
                                    model[b].error(0,y) = error;
                                    //std::cout << "ERROR: " << error << '\n';
                                }
                                else{
                                    error = model[b].error(0,y);
                                }
                                //sum += model[b].error(0,y);
                                //std::cout << "LAYER: " << model[b].type << ", ERROR " << error << '\n';
                                //std::cout << "ERROR PASSED!! " <<'\n';
                            }
                            else if(loss=="categorical_cross_entropy"){
                                /*
                                auto denum = ub-lb;
                                denum = denum <= EPSILON? EPSILON : denum;
                                auto y_ = std::fabs((model[b].output(0,y)-lb)/denum);
                                y_ = y_ <= EPSILON ? EPSILON : y_;
                                //y_ = model[b].output(0,y)>0?y_:-y_;
                                denum = (ub_target - lb_target);
                                denum = denum <= EPSILON? EPSILON : denum;
                                auto t = std::fabs((model[b].output_target(0,y)-lb_target)/denum);
                                t = t < EPSILON? EPSILON : t;
                                t = model[b].output_target(0,y)>0?t:-t;
                                error = (y_-t);
                                */
                                auto out_idx = model.size()-1;
                                //if(model[b].type=="output"){
                                if(b==out_idx){
                                    if(model[b].activation=="softmax" || model[b].activation=="sigmoid"){
                                        error = (model[b].output(0,y) - model[b].output_target(0,y));
                                        double denum = 2*(model[b].output(0,y));
                                        error /= (denum+EPS);
                                    }
                                    else{
                                        error = (model[b].output(0,y) - model[b].output_target(0,y));
                                        //double denum = (model[b].output(0,y) - std::pow(model[b].output(0,y),2));
                                        //error /= (denum + EPS);
                                    }
                                    //std::cout << "TARGET: " << model[b].output_target(0,y) << '\n';
                                    //model[b].error(0,y) = std::pow(error,2);
                                    model[b].error(0,y) = error;
                                    //std::cout << "ERROR: " << error << '\n';
                                }
                                else{
                                    error = model[b].error(0,y);
                                }
                            }
                            temp_error(0,y) = error;
                        }

                        /* ∂Output/∂Weight mixed up with ∂Error/∂Output */
                        for(int row=0; row<model[b].weight.rows(); row++){
                            for(int col=0; col<model[b].weight.cols(); col++){
                                if(model[b].activation=="leaky_relu"){
                                    double derr = model[b].output(0,col)>=0?1.0:0.01;
#ifdef ADAM_OPTIMIZER
                                    auto grad = derr*temp_error(0,col);

                                    auto g = grad;//batch_size;
                                    model[b].vs(row,col) = BETA*model[b].vs(row,col) + (1.0-BETA) * g;
                                    model[b].sqrs(row,col) = BETA_TWO*model[b].sqrs(row,col) + (1.0-BETA_TWO)*std::pow(g,2);

                                    auto v_bias_corr = model[b].vs(row,col)/(1.0-std::pow(BETA,ep+1));
                                    auto sqr_bias_corr =  model[b].sqrs(row,col)/(1.0-std::pow(BETA_TWO,ep+1));

                                    auto div = learning_rate * v_bias_corr/(std::sqrt(sqr_bias_corr)+EPS);

                                    model[b].weight(row,col) -= div;
#else
                                    auto delta = derr*(learning_rate*(temp_error(0,col))); //+ BETA*model[b].delta_weight(row,col);
                                    model[b].delta_weight(row,col) = delta;
                                    model[b].weight(row,col) += delta;
#endif
                                }
                                else if(model[b].activation=="sigmoid"){
                                    auto output = model[b].output(0,col);
                                    auto input = model[b-1].output(0,row);

#ifdef ADAM_OPTIMIZER
                                    auto grad = 2*(temp_error(0,col))*(output)*(1-output)*input;

                                    auto g = grad;//batch_size;
                                    model[b].vs(row,col) = BETA*model[b].vs(row,col) + (1.0-BETA) * g;
                                    model[b].sqrs(row,col) = BETA_TWO*model[b].sqrs(row,col) + (1.0-BETA_TWO)*std::pow(g,2);

                                    auto v_bias_corr = model[b].vs(row,col)/(1.0-std::pow(BETA,ep+1));
                                    auto sqr_bias_corr =  model[b].sqrs(row,col)/(1.0-std::pow(BETA_TWO,ep+1));

                                    auto div = learning_rate * v_bias_corr/(std::sqrt(sqr_bias_corr)+EPS);

                                    model[b].weight(row,col) -= div;
#else
                                    auto delta = learning_rate*2*(temp_error(0,col))*(output)*(1-output)*input + BETA*model[b].delta_weight(row,col);
                                    model[b].delta_weight(row,col) = delta;
                                    model[b].weight(row,col) += delta;
#endif
                                }
                                else if(model[b].activation=="softmax"){
                                    auto output = model[b].output(0,col);
                                    auto input = model[b-1].output(0,row);
                                    auto kronecker_delta = row==col?0:1;
#ifdef ADAM_OPTIMIZER
                                    auto grad = 2*(temp_error(0,col))*(output)*(kronecker_delta-output)*input;

                                    auto g = grad;//batch_size;
                                    model[b].vs(row,col) = BETA*model[b].vs(row,col) + (1.0-BETA) * g;
                                    model[b].sqrs(row,col) = BETA_TWO*model[b].sqrs(row,col) + (1.0-BETA_TWO)*std::pow(g,2);

                                    auto v_bias_corr = model[b].vs(row,col)/(1.0-std::pow(BETA,ep+1));
                                    auto sqr_bias_corr =  model[b].sqrs(row,col)/(1.0-std::pow(BETA_TWO,ep+1));

                                    auto div = learning_rate * v_bias_corr/(std::sqrt(sqr_bias_corr)+EPS);

                                    model[b].weight(row,col) -= div;
#else
                                    auto delta = learning_rate*2*(temp_error(0,col))*(output)*(kronecker_delta-output)*input; //+ BETA*model[b].delta_weight(row,col);
                                    model[b].delta_weight(row,col) = delta;
                                    model[b].weight(row,col) += delta;
#endif
                                }
                            }
                        }

                        //std::cout << "WEIGHT " <<b<<'('<< '0' <<' '<<'0'<<") "<<model[b].weight(0,0)<<'\n';
                        //model[b-1].output_target = model[b].output_target * pseudoinverse(model[b].weight);
                        model[b-1].error = model[b].error * model[b].weight.transpose();
                    }

                }

                /* ACCURACY TEST */
                auto end = std::chrono::steady_clock::now();

                auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
                int hour = 0;
                int minute = 0;
                int second = 0;
                second = time/1000;
                minute = second/60;
                hour = minute/60;
                second %= 60;
                minute %= 60;
                time %= 1000;
                std::stringstream ss;
                if(hour<10 && minute<10 && second <10){
                    ss << '0' << hour << ':' << '0' << minute  << ':' << '0' << second;//  << ':' << time;
                }
                else if(hour<10 && minute<10){
                    ss << '0' << hour << ':' << '0' << minute  << ':' << second;//  << ':' << time;
                }
                else if(hour<10 && second<10){
                    ss << '0' << hour << ':' << minute  << ':' << '0' << second;//  << ':' << time;
                }
                else if(hour<10){
                    ss << '0' << hour << ':' << minute  << ':' << second;//  << ':' << time;
                }
                else if(minute<10 && second <10){
                    ss << '0' << hour << ':' << '0' << minute  << ':' << '0' << second;//  << ':' << time;
                }
                else {
                    ss << hour << ':' << minute  << ':' << second;//  << ':' << time;
                }

                curr_time = ss.str();

                //int label_target_index = 0;
                //double maximum_target = Y[data][0];
                //for(int idx=0; idx<Y[data].size(); idx++){
                //    if(Y[data][idx]>maximum_target){
                //        label_target_index = idx;
                //        maximum_target = Y[data][idx];
                //    }
                //}

                //auto local_accuracy = model[model.size()-1].output(0,label_target_index);
                //accuracy_sum += local_accuracy;

                //local_accuracy = std::fabs((accuracy_sum/(data+1))*100.0);
                //if(local_accuracy>last_accuracy && !std::isnan(std::fabs(local_accuracy))){
                //    last_accuracy = local_accuracy;
                //    save_weight();
                //}
                //accuracy = accuracy==0.0?local_accuracy:accuracy;
                if(ep!=epoch-1||data!=X.size()-1){
                    //std::cout << "Accuracy: " << std::setprecision(7) << accuracy << "%, ";
                    std::cout << "Loss: " << std::setprecision(7) << losses*100.0 << "%, ";
                    std::cout << "Time Elapsed: "
                              << curr_time << "       " << '\r' <<std::flush;
                }
                //else{
                //    std::cout << "Accuracy: " << std::setprecision(7) << accuracy << "%, ";
                //    std::cout << "Time Elapsed: "
                //              << curr_time << '\n';
                //}
            }

            /* ACCURACY TEST */
            for(int data=0; data<X_.size(); data++){


                std::vector<double> Y_train;
                //auto trueval = 0;
                //auto truevalue = Y_[data][trueval];
                //for(int idx=0; idx<Y_[data].size(); idx++){
                //    if(Y_[data][idx]>truevalue){
                //        trueval = idx;
                //    }
                //}

                Y_train = predict(X_[data],true);

                //auto predval = 0;
                //auto predvalue = Y_train[predval];
                //for(int idx=0; idx<Y_train.size(); idx++){
                //    if(Y_train[idx]>predvalue){
                //        predval = idx;
                //    }
                //}

                //if(predval==trueval)
                //    predtrue++;

                auto tempred = 0.0;
                for(int idx=0; idx<Y_train.size(); idx++){
                    tempred += std::fabs(Y_train[idx]-Y_[data][idx]);
                }
                double ysize = Y_train.size();
                tempred/=ysize;

                predtrue += tempred;
            }
            double datasize = X_.size();
            //accuracy = (double)(predtrue/datasize);
            losses = (double)(predtrue/datasize);
            //std::cout << "ACCURACY TEST: " << accuracy << ", LAST ACCURACY: " << last_accuracy << "         " <<"\n";

            if(losses<=(last_loss+ACCURACY_TOLEARANCE) && !std::isnan(std::fabs(losses))){
                //last_loss = losses;
                save_weight();
            }

        }

        if(!std::isnan(std::fabs(losses))){
            last_loss = losses;
            save_weight();
        }

        //if(accuracy>=(last_accuracy-ACCURACY_TOLEARANCE) && !std::isnan(std::fabs(accuracy))){
        //    last_accuracy = accuracy;
        //    save_weight();
        //}
        //std::cout << "Accuracy: " << std::setprecision(7) << last_accuracy << "%, ";
        std::cout << "Loss: " << std::setprecision(7) << last_loss*100.0 << "%, ";
        std::cout << "Time Elapsed: "
                  << curr_time << "       " << '\n';


    }

    std::vector<double> predict(std::vector<double> X,bool istrain=false){

        std::vector<double> ret;

        assert(X.size()==model[0].output.cols());

        if(!istrain){
            load_weight();
            std::cout << "LOSS: " << last_loss*100.0 << '%' << '\n';
        }

        std::vector<Layer> model_predict = model;

        for(int idx=0; idx<model_predict.size(); idx++){
            if(model_predict[idx].type=="dropout"){
                model_predict.erase(model_predict.begin()+idx);
            }
        }

        for(int idx=0; idx<X.size(); idx++)
            model_predict[0].output(0,idx) = X[idx];

        for(int l=0; l<model_predict.size()-1; l++){
            if(model_predict[l+1].type=="batch_normalization"){

                auto mean = 0.0;
                for(int idx=0; idx<model_predict[l].output.cols(); idx++){
                    mean += model_predict[l].output(0,idx);
                }
                mean/=model_predict[l].output.cols();
                auto variance = 0.0;
                for(int idx=0; idx<model_predict[l].output.cols(); idx++){
                    variance += std::pow(model_predict[l].output(0,idx)-mean,2);
                }
                variance/=model_predict[l].output.cols();
                auto stddev = std::sqrt(variance+EPS);
                if(!model_predict[l+1].isFirstTrain){
                    model_predict[l+1].isFirstTrain = true;
                    model_predict[l+1].gamma = std::sqrt(variance);
                    model_predict[l+1].beta = mean;
                }
                for(int idx=0; idx<model_predict[l].output.cols(); idx++){
                    model_predict[l+1].X_cache(0,idx) = model_predict[l].output(0,idx);
                    auto xnorm = (model_predict[l].output(0,idx) - mean)/stddev;
                    model_predict[l+1].X_norm(0,idx) = xnorm;
                    xnorm = model_predict[l+1].gamma*xnorm+model_predict[l+1].beta;
                    model_predict[l].output(0,idx) = xnorm;
                }
            }
            else{
                model_predict[l+1].input = model_predict[l].output * model_predict[l+1].weight;
                if(model_predict[l+1].activation=="leaky_relu"){
                    model_predict[l+1].output = activation(model_predict[l+1].input);
                }
                else if(model_predict[l+1].activation=="sigmoid"){
                    model_predict[l+1].output = activation(model_predict[l+1].input,"sigmoid");
                }
                else if(model_predict[l+1].activation=="softmax"){
                    model_predict[l+1].output = activation(model_predict[l+1].input,"softmax");
                }
            }

        }

        for(int col=0; col<model_predict[model_predict.size()-1].output.cols(); col++){
            auto output = model_predict[model_predict.size()-1].output(0,col);
            ret.push_back(output);
            if(!istrain)
                std::cout << "[LABEL] "<<col<<": "<<output*100.0<<'%'<<'\n';
        }

        return ret;

    }

private:

    inline
    void save_weight(){
        std::ofstream out(weight_path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
        for(int idx =1; idx<model.size(); idx++){
            auto rows = model[idx].weight.rows();
            auto cols = model[idx].weight.cols();
            out.write((char*) (&rows), sizeof(rows));
            out.write((char*) (&cols), sizeof(cols));
            out.write((char*) model[idx].weight.data(), rows*cols*sizeof(Eigen::MatrixXd::Scalar));
        }
        //out.write((char*)(&last_accuracy),sizeof(last_accuracy));
        out.write((char*)(&last_loss),sizeof(last_loss));
        out.close();
    }

    inline
    void load_weight(){
        /* Must Create a Model*/
        if(model.size()==0){
            std::cout<<"Error! Must Create a Model!"<<'\n';
            return;
        }

        std::ifstream in(weight_path.c_str(), std::ios::in | std::ios::binary);
        for(int idx = 1; idx<model.size(); idx++){
            auto rows = model[idx].weight.rows();
            auto cols = model[idx].weight.cols();
            in.read((char*) (&rows),sizeof(rows));
            in.read((char*) (&cols),sizeof(cols));
            model[idx].weight.resize(rows, cols);
            in.read((char*)model[idx].weight.data(),rows*cols*sizeof(Eigen::MatrixXd::Scalar));
        }
        //in.read((char*)(&last_accuracy),sizeof(last_accuracy));
        in.read((char*)(&last_loss),sizeof(last_loss));
        in.close();
    }

    inline
    void randomize_weight(Eigen::MatrixXd &input, std::string initializer){


        if(initializer=="lecunn_uniform"){

            double fan_in = input.rows();
            auto limit = std::sqrt(3.0/fan_in);
            std::random_device rand{};
            std::mt19937 gen{rand()};
            std::uniform_real_distribution<double> randgen(-limit,limit);

            for(int row=0; row<input.rows(); row++){
                for(int col=0; col<input.cols(); col++){
                    input(row,col) = randgen(gen);
                }
            }

        }
        else if(initializer=="he_normal"){

            double fan_in = input.rows();
            auto limit = std::sqrt(2.0/fan_in);
            std::random_device rand{};
            std::mt19937 gen{rand()};
            std::normal_distribution<> randgen{0.0,limit};

            for(int row=0; row<input.rows(); row++){
                for(int col=0; col<input.cols(); col++){
                    input(row,col) = randgen(gen);
                }
            }

        }

    }

    inline double sum_error(Eigen::MatrixXd input){

        double ret = 0.0;

        for(int row=0; row<input.rows();row++)
            for(int col=0; col<input.cols();col++){
                //std::cout << "ERROR " << input(row,col) << '\n';
                ret += std::fabs(input(row,col));
            }

        return ret;

    }

    inline Eigen::MatrixXd activation(Eigen::MatrixXd input,std::string type="leaky_relu"){

        if(type=="leaky_relu"){
            /* Leaky ReLU */
            for(int col=0; col<input.cols(); col++){
                auto in = input(0,col);// + BIAS;
                in = in==0.0?in+BIAS:in;
                input(0,col) = std::max(0.01*in,in);
                //input(0,col) *= RELU_STEP;
            }
        }
        else if(type=="sigmoid"){
            /* Sigmoid */
            for(int col=0; col<input.cols(); col++){
                auto in = input(0,col)-BIAS;
                input(0,col) = 1.0/(1.0+std::exp(-(in)));
            }
        }
        else if(type=="softmax"){

            double sum = 0.0;
            double max = -INF;

            for(int col=0; col<input.cols(); col++){
                max = max<input(0,col)?input(0,col):max;
            }
            for(int col=0; col<input.cols(); col++){
                auto in = input(0,col) - max;
                sum += std::exp(in);
            }
            for(int col=0; col<input.cols(); col++){
                auto in = input(0,col) - max;
                input(0,col) = std::exp(in)/sum;
            }
        }

        return input;

    }

    inline
    BatchData divide_into_batch(BatchData data,int batch){

        BatchData ret;
        std::vector<std::pair<std::vector<double>,std::vector<double>>> datapair;
        auto X = data.first;
        auto Y = data.second;
        for(int idx=0; idx<X.size(); idx++){
            auto datatemp = std::make_pair(X[idx],Y[idx]);
            datapair.push_back(datatemp);

        }

       /*
        X.clear();
        Y.clear();

        std::random_shuffle(datapair.begin(),datapair.end());

        for(int idx=0; idx<datapair.size(); idx++){
            X.push_back(datapair[idx].first);
            Y.push_back(datapair[idx].second);
        }
        */


        std::vector<std::vector<double>> tempx, tempy;
        int size = X.size()/batch;

        for(int i=0; i<size; i++){
            std::random_device rd;
            auto lb = i*batch;
            auto ub = lb+batch-1;
            if(ub>=X.size()) ub = X.size()-1;
            std::uniform_int_distribution<int> rgen(lb,ub);
            int idx = rgen(rd);
            tempx.push_back(X[idx]);
            tempy.push_back(Y[idx]);
        }

        ret = std::make_pair(tempx,tempy);

        return ret;
    }

private:

    template <class MatT>
    inline
    Eigen::Matrix<typename MatT::Scalar, MatT::ColsAtCompileTime, MatT::RowsAtCompileTime>
    pseudoinverse(const MatT &mat, typename MatT::Scalar tolerance = typename MatT::Scalar{1e-4}) // choose appropriately
    {
        typedef typename MatT::Scalar Scalar;
        auto svd = mat.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        const auto &singularValues = svd.singularValues();
        Eigen::Matrix<Scalar, MatT::ColsAtCompileTime, MatT::RowsAtCompileTime> singularValuesInv(mat.cols(), mat.rows());
        singularValuesInv.setZero();
        for (unsigned int i = 0; i < singularValues.size(); ++i) {
            if (singularValues(i) > tolerance)
            {
                singularValuesInv(i, i) = Scalar{1} / singularValues(i);
            }
            else
            {
                singularValuesInv(i, i) = Scalar{0};
            }
        }
        return svd.matrixV() * singularValuesInv * svd.matrixU().adjoint();
    }

private:
    std::vector<Layer> model;
    int epoch;
    int batch_size;
    double learning_rate;
    std::string weight_path;
    double last_accuracy;
    double last_loss;

};


}



#endif //ANN_HPP
