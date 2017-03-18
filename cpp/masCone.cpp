//
// Created by palsol on 29.01.17.
//

#include "../include/masCone.h"

using namespace std;

masCone::masCone() {
    //initialization(test);
}

masCone::masCone(vector<double> coneData) {
    initialization(coneData);
}

masCone::~masCone() {
}

int masCone::getDim() {
    return order.size();
}

double masCone::distance(vector<double> A, vector<double> B) {
    double result = 0;
    if (A.size() == B.size()) {
        for (int i = 0; i < A.size(); i++) {
            result += pow(B[i] - A[i], 2);
        }
    }
    return result;
}


vector<double> masCone::approximation(vector<double> signal) {

    vector<double> signalApproximation;
    if (signal.size() == getDim()) {

        signalApproximation.push_back(signal[order[0]]);
        for (int i = 1; i < signal.size() && i < getDim(); i++) {
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
    }
    return restore(signalApproximation);
}

vector<double> masCone::getChunkProximity(vector<double> signal) {

    vector<double> proximities;

    if (getDim() < signal.size()) {
        cout << "start!" << endl;
        for (int i = 0; i < signal.size() - getDim(); i++) {
            proximities.push_back(getProximity(vector<double>(signal.begin() + i, signal.begin() + i + getDim())));
        }
    }
    return proximities;
}


double masCone::getProximity(vector<double> signal) {

    double result = 0;
    if (signal.size() == getDim()) {
        vector<double> signalApproximation = approximation(signal);
        result = distance(signal, signalApproximation);
    }
    return result;
}

double masCone::getProximityConstDistinction(vector<double> signal) {

    double result = 0;
    if (signal.size() == getDim()) {
        vector<double> signalApproximation = approximation(signal);
        vector<double> signalConstApproximation;
        double sum = std::accumulate(signal.begin(), signal.end(), 0.0);
        double mean = sum / signal.size();
        signalConstApproximation.assign(signal.size(), mean);

        result = distance(signal, signalApproximation) / distance(signalConstApproximation, signalApproximation);
    }
    return result;
}

void masCone::initialization(vector<double> data) {

    order.resize(data.size());
    iota(order.begin(), order.end(), 0);

    sort(order.begin(), order.end(),
         [&data](size_t i1, size_t i2) { return data[i1] < data[i2]; });
}

vector<double> masCone::restore(vector<double> data) {

    vector<double> restored;
    restored.resize(order.size());

    for (int i = 0; i < order.size(); i++) {
        restored[order[i]] = data[i];
    }

    return restored;
}
