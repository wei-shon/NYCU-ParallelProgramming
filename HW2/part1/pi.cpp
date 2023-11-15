#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <pthread.h>
using namespace std;

long long number_in_circle; // Use atomic to avoid data race
pthread_mutex_t locker = PTHREAD_MUTEX_INITIALIZER;

void* monteCarloThread(void* arg) {
    int min = -1;
    int max = 1;
    long long local_count = 0;
    unsigned int seed =  time(NULL);
    long long tosses = (long long) arg;
    for (long long toss = 0; toss < tosses; toss++) {
        double x = (max - min) * (rand_r(&seed) / (RAND_MAX + 1.0)) + min;
        double y = (max - min) * (rand_r(&seed) / (RAND_MAX + 1.0)) + min;
        double distance_squared = x * x + y * y;
        if (distance_squared <= 1)
            local_count++;
    }
    pthread_mutex_lock(&locker);
    number_in_circle += local_count;
    pthread_mutex_unlock(&locker);
    return 0;
}

int main(int argc, char* argv[]) {
    int threadsNum = stoi((string)argv[1]);
    long long number_of_tosses = stoll((string)argv[2]);

    number_in_circle = 0;

    // vector<pthread_t> threads;
    pthread_t threads[threadsNum];
    for (int i = 0; i < threadsNum; i++) {
        pthread_create(&threads[i], NULL, monteCarloThread,(void*) (number_of_tosses / threadsNum));
    }

    for (int i = 0; i < threadsNum; i++) {
        pthread_join(threads[i], NULL);
    }

    double pi_estimate = 4 * (static_cast<double>(number_in_circle) / number_of_tosses);
    cout << pi_estimate << endl;

    return 0;
}