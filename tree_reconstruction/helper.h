#ifndef HELPER_H
#define HELPER_H
#include <list>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <stdlib.h>   
#include <time.h>     
#include <set>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <utility>
#include <algorithm>
#include <limits.h>
#include <unistd.h>
#include <sys/stat.h>
#include <chrono>
#include <pthread.h>
typedef unsigned long long ulong_t;
typedef long long long_t;

#define INF 0x3f3f3f3f


using namespace std;

#define LOC	cout << ((*(new string(">> @"))).append(__FILE__).append("::").append(to_string(__LINE__)).append("::").append(__FUNCTION__).append("() - "))
//#define LOC	((*(new string("@"))).append(__FILE__).append("::").append(to_string(__LINE__)).append("::").append(__FUNCTION__).append("() - "))

/* Helper function to split the fileLine */
void split(const string &s, char delim, vector<string> &elems);
/* Function to split the fileLine using specified delimiter */
vector<string> split(const string &s, char delim);
bool file_exists (const std::string& name);
size_t file_size (const std::string& name);


#endif /* GRAPH_H */
