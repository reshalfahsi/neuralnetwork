#include <iostream>

/* TESTING */
#include "../ann/ann.hpp"
#include "../ann/csvreader.hpp"

#define TEST 100.0
#define TRAINING 10.0

//#define DUMMY

#ifndef DUMMY
#define BATCH 30
#define EPOCH 15
#define LEARNING_RATE 0.001
#else
#define BATCH 3
#define EPOCH 150
#define LEARNING_RATE 0.001
#endif


void coba_ann(bool isTrain=true){

#ifndef DUMMY
    std::string csv_path = std::string("mnist/mnist_test.csv");
    std::string weight_path = std::string("mnist/saved_weight/weight.dat");
#else
    std::string csv_path = std::string("mnist/mnist_dummy.csv");
    std::string weight_path = std::string("mnist/saved_weight/weight_dummy.dat");

    //std::string csv_path = std::string("mnist_dummy_10.csv");
    //std::string weight_path = std::string("saved_weight/weight_dummy_10.dat");
#endif

    Data X,Y;
    CSVReader train_data(csv_path);
    auto datalist = train_data.getData();
    //std::cout << "DEBUGGING: " << datalist.size() << std::endl;

    for(auto vec : datalist)
    {
        int idx = 0;
        std::vector<double> data_temp;
        for(auto data : vec)
        {
            if(idx==0){
                if(data=="0"){
                    Y.push_back({1,0,0,0,0,0,0,0,0,0});
                }
                else if(data=="1"){
                    Y.push_back({0,1,0,0,0,0,0,0,0,0});
                }
                else if(data=="2"){
                    Y.push_back({0,0,1,0,0,0,0,0,0,0});
                }
                else if(data=="3"){
                    Y.push_back({0,0,0,1,0,0,0,0,0,0});
                }
                else if(data=="4"){
                    Y.push_back({0,0,0,0,1,0,0,0,0,0});
                }
                else if(data=="5"){
                    Y.push_back({0,0,0,0,0,1,0,0,0,0});
                }
                else if(data=="6"){
                    Y.push_back({0,0,0,0,0,0,1,0,0,0});
                }
                else if(data=="7"){
                    Y.push_back({0,0,0,0,0,0,0,1,0,0});
                }
                else if(data=="8"){
                    Y.push_back({0,0,0,0,0,0,0,0,1,0});
                }
                else if(data=="9"){
                    Y.push_back({0,0,0,0,0,0,0,0,0,1});
                }
            }
            else{
                double data_norm = std::atof(data.c_str());
                data_temp.push_back(data_norm/255.0);
            }
            idx++;
        }
        X.push_back(data_temp);
        data_temp.clear();
    }

    fukuro::ANN myNetwork(EPOCH,LEARNING_RATE,BATCH,weight_path);

#ifdef DUMMY
    myNetwork.add_layer("input",784,"leaky_relu");
    myNetwork.add_layer("hidden",300,"leaky_relu");
    //myNetwork.add_layer("batch_normalization");
    //myNetwork.add_layer("dropout",0.2);
    myNetwork.add_layer("hidden",300,"leaky_relu");
    //myNetwork.add_layer("dropout",0.2);
    //myNetwork.add_layer("batch_normalization");
    myNetwork.add_layer("output",10,"softmax");
#else
    myNetwork.add_layer("input",784,"leaky_relu");
    myNetwork.add_layer("hidden",300,"leaky_relu");
    myNetwork.add_layer("hidden",300,"leaky_relu");
    myNetwork.add_layer("output",10,"softmax");
#endif
    if(isTrain){
        auto sum_correct =0;
        myNetwork.fit(X,Y,"categorical_cross_entropy","he_normal");

        for(int iter=0; iter<TRAINING;iter++){

            std::random_device rd;
            std::uniform_int_distribution<int> rgen(0,X.size()-1);
            int idx = rgen(rd);

            std::vector<double> x = X[idx],y_test = Y[idx];

            auto y = myNetwork.predict(x);
            int pred = 0;
            double biggest = y[pred];
            for(int i=0; i<y.size(); i++){
                if(biggest<y[i]){
                    biggest = y[i];
                    pred = i;
                }
            }
            int test = 0;
            biggest = y_test[test];
            for(int i=0; i<y_test.size(); i++){
                if(biggest<y_test[i]){
                    biggest = y_test[i];
                    test = i;
                }
            }
            std::cout << "Predicted : " << pred << '\n';
            std::cout << "True Value : " << test << '\n';
            if(pred==test)
                sum_correct++;
            std::cout << "Predicted Correct: " << sum_correct << ' ' << "times" <<'\n';
        }
        double accuracy = (sum_correct/TRAINING)*100.0;
        std::cout << "Predicted Accuracy: " << accuracy << '%' <<'\n';
    }
    else{
        auto sum_correct =0;

        std::cout << "TESTING" << '\n';
        std::cout << "<<----------------------------------------------------" << '\n';

        for(int iter=0; iter<TEST; iter++){
            std::random_device rd;
            std::uniform_int_distribution<int> rgen(0,X.size()-1);
            int idx = rgen(rd);

	    //std::cout << "DEBUGGING: " << X.size() << std::endl;	    
		
            std::vector<double> x = X[idx],y_test = Y[idx];

	    //std::cout << "DEBUGGING" << std::endl;

            auto y = myNetwork.predict(x);
            int pred = 0;
            double biggest = y[pred];
            for(int i=0; i<y.size(); i++){
                if(biggest<y[i]){
                    biggest = y[i];
                    pred = i;
                }
            }
            int test = 0;
            biggest = y_test[test];
            for(int i=0; i<y_test.size(); i++){
                if(biggest<y_test[i]){
                    biggest = y_test[i];
                    test = i;
                }
            }
            std::cout << "Predicted : " << pred << '\n';
            std::cout << "True Value : " << test << '\n';
            if(pred==test)
                sum_correct++;
            std::cout << "---------------------------------------------------->>" << '\n';
        }
        std::cout << "Predicted Correct: " << sum_correct << ' ' << "times" <<'\n';
        double accuracy = (sum_correct/TEST)*100.0;
        std::cout << "Predicted Accuracy: " << accuracy << '%' <<'\n';
    }

}

//*/
int main(){

#ifndef DUMMY
    std::cout << "ANN Example" << '\n';
    bool isTrain=false;
#else
    std::cout << "anN DuMMy ExaMpLE" << '\n';
    bool isTrain=false;
#endif

    coba_ann(isTrain);

}

