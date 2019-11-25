/*
    Ben Vecchio
    10/4/2019
    CISC 4631
    HW1
*/
#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>
#include "message.h"
using namespace std;

/* Reads data from .csv file and returns it as a vector of objects */
vector<message> getSetOfData(string file)
{
    vector<message> data;
    ifstream source(file);

    /* Ignores first line of file */
    string dummy;
    getline(source,dummy);

    /* Iterates through each line of file */
    while(source.good())
    {
        string id, feature, label;
        message newMessage;

        /* Assigns ID of message */
        getline(source, id, ',');
        newMessage.id = id;

        /* Assigns features f1 through 57 */
        for (int i = 0; i < 57; i++)
        {
            getline(source, feature, ',');
            newMessage.features[i] = stod(feature);
        }

        /* Assigns label of message */
        getline(source, label);
        newMessage.label = stoi(label);

        /* Pushes object to set */
        data.push_back(newMessage);
    }
    return data;
}

/*
    Gets average value of each feature (f1-f57) and returns values in vector of doubles
    For example, the first value in the vector will be the average value of feature f1 in the training set
*/
vector<double> getAverages(vector<message> training_data)
{
    vector<double> averages;

    /* Iterates through each feature (e.g. f1, f2, f3, ... f57) */
    for (int i = 0; i < 57; i++)
    {
        double sum = 0;

        /* Adds feature #i from each training datum to the sum*/
        for (int j = 0; j < training_data.size(); j++)
        {
            sum += training_data[j].features[i];
        }

        /* Divides sum by number of data and pushes it to the set */
        averages.push_back(sum / training_data.size());
    }
    return averages;
}

/* Gets standard deviation for each average computed in getAverages() and returns values in vector of doubles */
vector<double> getStdDevs(vector<message> training_data, vector<double> averages)
{
    vector<double> stdDevs;

    /* Iterates through each feature (e.g. f1, f2, f3, ... f57) */
    for (int i = 0; i < 57; i++)
    {
        /*
            Formula for variance: (sum( f(i)^2 ) / n) - average(f)
                f = a given feature
                f(i) = occurrence #i of feature f
                average(f) = average value of feature f across training set
        */
        double sum = 0;
        int n = training_data.size();
        for (int j = 0; j < n; j++)
        {
            sum += training_data[j].features[i] * training_data[j].features[i];
        }
        double variance = (sum/n) - (averages[i]*averages[i]);

        /* Standard deviation is computed as square root of variance */
        stdDevs.push_back(sqrt(variance));
    }
    return stdDevs;
}

/* Z-score normalization of set of data */
vector<message> zScoreNormalization(vector<message> data, vector<double> averages, vector<double> stdDevs)
{
    /* Iterates through each datum in set */
    for (int i = 0; i < data.size(); i++)
    {
        /* Iterates through each feature (e.g. f1, f2, f3, ... f57) of datum #i */
        for (int j = 0; j < 57; j++)
        {
            /*
                Formula for z-score: ( f-µ(f) )/ σ(f)
                    f = a given feature
                    f(i) = occurrence #i of feature f
                    µ(f) = average value of feature f
                    σ(f) = standard deviation of µ(f)
            */
            data[i].features[j] = (data[i].features[j] - averages[j]) / stdDevs[j];
        }
    }
    return data;
}

/* Gets distance between two data */
double getDistance(message x, message y)
{
    /*
        Formula for distance: sqrt( sum ( x(i)-y(i) ) )
            x(i) = feature #i of message x
            y(i) = feature #i of message y
        Based off the formula for Euclidean distance
    */
    double sum = 0;
    for (int i = 0; i < 57; i++)
    {
        sum += (x.features[i] - y.features[i])*(x.features[i] - y.features[i]);
    }
    return sqrt(sum);
}

/* Sorts training data by each datum's distance from message m */
vector<message> sortTrainingDataByDistance(vector<message> training_data, message m)
{
    for (int i = 0; i < training_data.size(); i++)
    {
        /* Gets training datum's distance from message m */
        training_data[i].distance_from_test_datum = getDistance(training_data[i], m);
    }

    /*
        Sorts training data by distance_from_test_datum attribute and returns modified set
        This attribute is only ever referenced or modified in this function
    */
    sort(training_data.begin(), training_data.end());
    return training_data;
}

/* Predicts label of message m by comparing it to the k closest objects in training set */
int predictLabel(message m, int k, vector<message> training_data)
{
    /* Vote tallys for "spam" or "not spam" */
    int spam = 0, no = 0;

    /* Sorts training data by distance from message m */
    vector<message> sorted_by_distance = sortTrainingDataByDistance(training_data, m);

    /*
        We only need to look the k closest training data
        Fortunately, the training data is sorted by distance
        So we just need to iterate through the first k objects
    */
    for (int i = 0; i < k; i++)
    {
        /* If it's spam, vote yes; if not, vote no */
        if (sorted_by_distance[i].label == 1)
            spam++;
        else
            no++;
    }

    /*
        Results of poll
        We don't have to worry about a tie because k is always an odd number
    */
    return (spam > no) ? 1 : 0;
}

/* Compares label predictions to actual label values of test data and returns percentage of correct predictions as decimal */
double accuracy(int k, vector<message> test_data, vector<message> training_data)
{
    double total = 0, correct = 0;
    for (int i = 0; i < test_data.size(); i++)
    {
        // cout << "Processing message " << test_data[i].id << endl;
        if (test_data[i].label == predictLabel(test_data[i], k, training_data))
            correct++;
        total++;
    }
    return correct/total;
}

int main()
{
    vector<message> training_data = getSetOfData("spam_train.csv");
    vector<message> test_data = getSetOfData("spam_test.csv");
    int k[10] = {1, 5, 11, 21, 41, 61, 81, 101, 201, 401};

    /* Question 1a */
    for (int i = 0; i < 10; i++)
    {
        cout << "k=" << k[i] << " ==> accuracy of " << accuracy(k[i], test_data, training_data) << endl;
    }
    cout << endl;

    /* Question 1b */
    vector<double> averages = getAverages(training_data);
    vector<double> stdDevs = getStdDevs(training_data, averages);
    training_data = zScoreNormalization(training_data, averages, stdDevs);
    test_data = zScoreNormalization(test_data, averages, stdDevs);
    cout << "Normalized" << endl;
    for (int i = 0; i < 10; i++)
    {
        cout << "k=" << k[i] << " ==> accuracy of " << accuracy(k[i], test_data, training_data) << endl;
    }
    cout << endl;

    /* Question 1c */
    vector<message> testing(test_data.begin(), test_data.begin()+50);
    for (int i = 0; i < testing.size(); i++)
    {
        cout << testing[i].id << " ";
        for (int j = 0; j < 10; j++)
        {
            cout << ( (predictLabel(testing[i], k[j], training_data)==1) ? "spam" : "no" ) << (j < 9 ? ", " : "");
        }
        cout << endl;
    }
    return 0;
}
