#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <climits>

using namespace std;

class LogisticRegression {
public:
    LogisticRegression() : coefficients_({}) {}
    vector<double> coefficients() {return coefficients_;}
    
    void Fit(const vector<vector<double> >& trainingSamples,const vector<double>& labels) {
        // IMPLEMENT ME
        if (trainingSamples.size() == 0){
            //TODO:
            //ERROR, sample is null;
            // return;
        }
        coefficients_.resize(trainingSamples[0].size()+1, 0.0);
        vector<double> w_old(trainingSamples[0].size()+1, 1.0);

        while (myDistance(w_old, coefficients_)>0.0001){
            
            w_old=coefficients_;
            int randNum = rand()%labels.size();

            coefficients_[0] = w_old[0] + LearningRate()*(labels[randNum] - Predict(trainingSamples[randNum]));
            
            for (int i = 1; i < coefficients_.size(); i++){
                coefficients_[i] = w_old[i] + LearningRate()*(labels[randNum] - Predict(trainingSamples[randNum]))*trainingSamples[randNum][i-1];
            }
          
        }
        
    }
    
    
    double LearningRate() {
        return 0.01;
    }
    
    double Predict(const vector<double>& sample) {
        // IMPLEMENT ME
        double mySum = coefficients_[0];
        for (int i = 1; i < coefficients_.size(); i++){
            mySum += coefficients_[i] * sample[i-1];
        }
        
        return 1.0/(1.0+exp(-mySum)) ;
    }
    
protected:
    vector<double> coefficients_;
    
private:
    
    double myDistance(vector<double> w_old, vector<double> w_new){
        double difSum = 0.0;
        for (int i  = 0; i < w_old.size(); i++){
            difSum += (w_old[i] - w_new[i]) * (w_old[i] - w_new[i]);
        }
        
        return sqrt(difSum);
    }
};


class L2LogisticRegression: public LogisticRegression {
public:
    L2LogisticRegression(double l2Penalty) : l2Penalty_(l2Penalty) {}
    
    void Fit(const vector<vector<double> >& trainingSamples, const vector<double>& labels) {
        // IMPLEMENT ME
        if (trainingSamples.size() == 0){
            //TODO:
            //ERROR, sample is null;
            // return;
        }
        eta = 0.01;
        steps = 0;
        coefficients_.resize(trainingSamples[0].size()+1, 0.0);
        vector<double> w_old(trainingSamples[0].size()+1, 1.0);
        while (myDistance(w_old, coefficients_)>0.0001){
            w_old=coefficients_ ;
            double curEta = LearningRate(steps);
            int randNum = rand()%labels.size();
            
            coefficients_[0] = (1.0 - curEta * l2Penalty_) * w_old[0] + curEta * (labels[randNum] - Predict(trainingSamples[randNum]));
            
            for (int i = 0; i < coefficients_.size()-1; i++){
                coefficients_[i+1] = (1.0 - curEta * l2Penalty_) * w_old[i+1] + curEta * (labels[randNum] - Predict(trainingSamples[randNum]))*trainingSamples[randNum][i];
            }
            steps++;
        }
    }
    
    double LearningRate(int iterCount) {
        // IMPLEMENT ME
        eta = eta * 1.0 / (1.0 + eta * l2Penalty_ * steps);
        return eta;
    }
    
    double Predict(const vector<double>& sample) {
        // IMPLEMENT ME
        double mySum = coefficients_[0];
        
        for (int i = 1; i < coefficients_.size(); i++){
            mySum += coefficients_[i] * sample[i-1];
        }
        
        return 1.0/(1.0+exp(-mySum)) ;
    }
    
private:
    double l2Penalty_;
    double eta;
    int steps;
    
    double myDistance(vector<double> w_old, vector<double> w_new){
        double difSum = 0.0;
        for (int i  = 0; i < w_old.size(); i++){
            difSum += (w_old[i] - w_new[i]) * (w_old[i] - w_new[i]);
        }
        
        return sqrt(difSum);
    }
};

#ifndef __main__
#define __main__

int main() {
    LogisticRegression logisticRegression;
    
    vector<vector<double> > trainingSamples = {
        { 1.0, 2.3, 2.2 },
        { 0.3, 4.3, 2.1 },
        { 3.2, 0.0, 2.2 },
        { 2.3, -1.3, 2.3 },
        { 1.3, 3.2, 2.0 }
    };
    vector<double> labels = { 1.0, 1.0, 0.0, 0.0, 1.0 };
    logisticRegression.Fit(trainingSamples, labels);
    
    vector<vector<double> > testingSamples = {
        { -1.3, 3.2, 2.0 },
        { 3.3, -1.2, 2.0}
    };
    
    // should print 1.0, 0.0
    for (vector<double> testingSample : testingSamples) {
        double prediction = logisticRegression.Predict(testingSample);
        if (prediction>0.5)
            cout << "1.0" << "\t";
        else cout<< "0.0" <<endl;
    }
    
    L2LogisticRegression l2LogisticRegression(0.1);
    
    trainingSamples = {
        { 1.0, 2.3, 2.2 },
        { 0.3, 4.3, 2.1 },
        { 3.2, 0.0, 2.2 },
        { 2.3, -1.3, 2.3 },
        { 1.3, 3.2, 2.0 }
    };
    labels = { 1.0, 1.0, 0.0, 0.0, 1.0 };
    l2LogisticRegression.Fit(trainingSamples, labels);
    
    testingSamples = {
        { -1.3, 3.2, 2.0 },
        { 3.3, -1.2, 2.0}
    };
    
    // should print 1.0, 0.0
    for (vector<double> testingSample : testingSamples) {
        double prediction = l2LogisticRegression.Predict(testingSample);
        if (prediction>0.5)
            cout << "1.0" << "\t";
        else cout<< "0.0" <<endl;
    }
    return 0;
}

#endif
