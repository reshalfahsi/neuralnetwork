#include "../ann/ann.hpp"
#include "../ann/csvreader.hpp"

#include <map>

#define TEST 100.0
#define TRAINING 10.0

#define BATCH 1
#define EPOCH 10
#define LEARNING_RATE 0.001


void house_price(bool isTrain=true){
    /*
     * "Index","Living Space","Beds","Baths","Zip","Year","List Price"
     *
     */

    std::map<std::string,int> features;

    std::string csv_path = std::string("houseprice/houseprice.csv");
    std::string weight_path = std::string("houseprice/saved_weight/weight.dat");

    Data X,Y;
    CSVReader train_data(csv_path);
    auto datalist = train_data.getData();

    int size = datalist[0].size();
    std::vector<int> find_max(size,-INF);

    int index = 0;
    for(auto vec : datalist)
    {
        int idx = 0;
        std::vector<double> data_temp;
        for(auto data : vec)
        {
            if(index==0){
                features[data] = idx;
                //std::cout << "DATA: " << data << ", INDEX: " << idx << '\n';
            }
            else{
                if( idx==features["Index"] || idx==features["Zip"] || idx==features["Year"] ){
                    idx++;
                    continue;
                }
                else if(idx==features["List Price"]){
                    double temp = std::atof(data.c_str());
                    find_max[idx]= find_max[idx]<temp? temp:find_max[idx];
                    Y.push_back({temp});
                }
                else{
                    //std::cout << "HELLO, WORLD!" << '\n';
                    double temp = std::atof(data.c_str());
                    find_max[idx]= find_max[idx]<temp? temp:find_max[idx];
                    data_temp.push_back(temp);
                }
            }
            idx++;
        }
        if(data_temp.size()!=0)
            X.push_back(data_temp);
        data_temp.clear();
        index++;
    }

    for(int data=0; data<X.size();data++){
        for(int idx=0; idx<X[data].size();idx++){
            X[data][idx] /= find_max[idx];
        }
    }

    for(int data=0; data<Y.size();data++){
        for(int idx=0; idx<Y[data].size();idx++){
            Y[data][idx] /= find_max[find_max.size()-1];
            //std::cout << Y[data][idx] << std::endl;
        }
    }

    //std::cout << "X size: " << X.size() << ", Y size: " << Y.size() << '\n';

    int input_node = X[0].size();
    fukuro::ANN myNetwork(EPOCH,LEARNING_RATE,BATCH,weight_path);

    myNetwork.add_layer("input",input_node,"sigmoid");
    myNetwork.add_layer("hidden",500,"sigmoid");
    myNetwork.add_layer("batch_normalization");
    myNetwork.add_layer("hidden",500,"sigmoid");
    //myNetwork.add_layer("dropout",0.2);
    //myNetwork.add_layer("hidden",90,"sigmoid");
    //myNetwork.add_layer("dropout",0.2);
    myNetwork.add_layer("output",1,"sigmoid");

    if(isTrain){

        myNetwork.fit(X,Y,"mse","lecunn_uniform");

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<> randgen(0,input_node);
        index = randgen(gen);

        auto pred = myNetwork.predict(X[index]);
        auto trueval = Y[index];

        double sum_error = 0.0;
        double data_sum = 0.0;
        for(int i; i<trueval.size(); i++){
            data_sum += trueval[i];
            sum_error += std::fabs(pred[i]-trueval[i]);
        }

        double accuracy = (1.0 - (sum_error/data_sum));
        std::cout << "ACCURACY: " << accuracy*100.0 << '%' << '\n';
        std::cout << "REAL PRICE: $" << trueval[0]*find_max[find_max.size()-1] << '\n';
        std::cout << "PREDICTED PRICE: $" << pred[0]*find_max[find_max.size()-1] << '\n';

    }
    else{

        std::random_device rd{};
        std::mt19937 gen{rd()};
        std::uniform_int_distribution<> randgen(0,input_node);
        index = randgen(gen);

        auto pred = myNetwork.predict(X[index]);
        auto trueval = Y[index];

        double sum_error = 0.0;
        double data_sum = 0.0;
        for(int i; i<trueval.size(); i++){
            data_sum += trueval[i];
            sum_error += std::fabs(pred[i]-trueval[i]);
        }

        double accuracy = (1.0 - (sum_error/data_sum));
        std::cout << "ACCURACY: " << accuracy*100.0 << '%' << '\n';

    }

}


int main(){

    std::cout << "House Price ~ Regression Example" << '\n';

    bool istrain = true;

    house_price(istrain);

}
