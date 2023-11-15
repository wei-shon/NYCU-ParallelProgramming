#include<iostream>
#include <cstdlib> /* 亂數相關函數 */
#include <ctime>   /* 時間相關函數 */

using namespace std;

int main(int argc, char *argv[]){


    // cout<<argv[0]<<endl;
    // cout<<argv[1]<<endl;

    int threadsNum = stoi( (string) argv[1]);
    long long int number_of_tosses = stoll( (string) argv[2]);

    int min = -1;
    int max = 1;
    srand( time(NULL) );

    // cin>>number_of_tosses;
    long long int number_in_circle = 0;

    for (long long int toss = 0; toss < number_of_tosses; toss ++) {
        double x = (max - min) * (rand() / (RAND_MAX + 1.0)) + min;
        double y = (max - min) * (rand() / (RAND_MAX + 1.0)) + min;
        // cout<<x<<" "<<y<<endl;
        double distance_squared = x * x + y * y;
        if ( distance_squared <= 1)
            number_in_circle++;
    }
    double pi_estimate = 4 * (number_in_circle /(( double ) number_of_tosses));
    cout<<pi_estimate<<endl;
    return 0;
}