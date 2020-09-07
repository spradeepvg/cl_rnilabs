#include "helper.h"
/* Helper function to split the fileLine */
void split(const string &s, char delim, vector<string> &elems) 
{
        stringstream ss;
        ss.str(s);
        string item;
        while (getline(ss, item, delim))
        {
                elems.push_back(item);
        }
}

/* Function to split the fileLine using specified delimiter */
vector<string> split(const string &s, char delim)
{
        vector<string> elems;
        split(s, delim, elems);
        return elems;
}


bool file_exists (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

size_t file_size (const std::string& name) {
  struct stat buffer;   
  if(stat (name.c_str(), &buffer) != 0) return 0;
  return buffer.st_size;
}
