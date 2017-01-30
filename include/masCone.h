//
// Created by palsol on 29.01.17.
//
#pragma once

#ifndef MASLIB_MASCONE_H
#define MASLIB_MASCONE_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>
#include <numeric>
#include <random>
#include <math.h>
using namespace std;

class masCone{

public:
    masCone();
    masCone(vector<double> coneData);
    ~masCone();
    int getDim();
    vector<int> order;
    vector<double> approximation(vector<double> signal);
    vector<double> getChunkProximity(vector<double> signal);
    double getProximity(vector<double> signal);

    static double distance(vector<double> A, vector<double> B);

private:
    void initialization(vector<double> data);
    vector<double> restore(vector<double> data);
};

#endif //MASLIB_MASCONE_H
