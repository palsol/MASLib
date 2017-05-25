//
// Created by palsol on 02.02.17.
//

#include <iostream>
#include "include/masCone.h"

using namespace std;

double distance(vector<double> A, vector<double> B);
vector<double> approximation(vector<double> signal);
double simpleMorfComparison(vector<double> a, vector<double> b);
vector<double> restore(vector<double> data);

int main(){

    vector<double> a = {1, 2, 3, 4};
    vector<double> b = {1, 4, 2, 2};
    double result = simpleMorfComparison(a, b);
    cout<< result << endl;
}

double distance(vector<double> A, vector<double> B) {
    double result = 0;
    if (A.size() == B.size()) {
        for (int i = 0; i < A.size(); i++) {
            result += pow(B[i] - A[i], 2);
        }
    }
    return result;
}

vector<double> restore(vector<double> data, vector<int> order) {

    vector<double> restored;
    restored.resize(order.size());

    for (int i = 0; i < order.size(); i++) {
        restored[order[i]] = data[i];
    }

    return restored;
}

vector<double> approximation(vector<double> signal, vector<int> order) {

    vector<double> signalApproximation;

    signalApproximation.push_back(signal[order[0]]);
    for (int i = 1; i < signal.size() && i < order.size(); i++) {
        double aprox = signal[order[i]];
        int j = 0;

        for (j = 0; i > j && aprox < signalApproximation[i - j - 1]; j++) {
            aprox = (aprox * (j + 1) + signal[order[i - j - 1]]) / (j + 2);
        }

        signalApproximation.push_back(aprox);
        for (int k = 1; k <= j; k++) {
            signalApproximation[i - k] = aprox;
        }
    }

    return restore(signalApproximation, order);
}


double simpleMorfComparison(vector<double> a, vector<double> b){

    vector<int> order;

    order.resize(a.size());
    iota(order.begin(), order.end(), 0);

    sort(order.begin(), order.end(),
         [&a](size_t i1, size_t i2) { return a[i1] < a[i2]; });


    vector<double> signalApproximation = approximation(b, order);
    vector<double> signalConstApproximation;
    double sum = std::accumulate(b.begin(), b.end(), 0.0);
    double mean = sum / b.size();
    signalConstApproximation.assign(b.size(), mean);

    return distance(b, signalApproximation) / distance(signalConstApproximation, signalApproximation);

}