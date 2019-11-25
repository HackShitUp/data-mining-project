#include <vector>
using namespace std;

class message
{
    public:
        message() {}
        string id;
        double features[57];
        int label;
        double distance_from_test_datum; // only used when determining label of test data
        bool operator < (const message &other) const {
            return distance_from_test_datum < other.distance_from_test_datum;
        }
};
