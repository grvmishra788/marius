#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>

using namespace std;
# define ll unsigned long long int

vector<pair<int, int> > decodeNodes(vector<pair<string, string> > list)
{
    vector<pair<int, int> > decodedList;
    ll base = (ll) pow(10, 10);
    for (int i = 0; i < list.size(); i++)
    {
        char *stopstring;
        char *node1 = const_cast<char *>(list[i].first.c_str());
        char *node2 = const_cast<char *>(list[i].second.c_str());

        ll decoded_node1 = strtoull(node1,&stopstring,10) %  base;
        ll decoded_node2 = strtoull(node2,&stopstring,10) %  base;

        decodedList.push_back(make_pair(decoded_node1, decoded_node2));
    }

    return decodedList;
}

int main()
{
    // Read the contents of the updated edges file
    ifstream input_file;
    input_file.open("updated_edges.csv");

    if (!input_file.is_open())
    {
        cout << "Error opening the updated edges file" << endl;
        return 0;
    }

    vector<pair<string, string> > edges_list;
    string node1, node2;
    
    while (std::getline(input_file, node1, ',') && std::getline(input_file, node2))
    {
        edges_list.push_back(make_pair(node1, node2));
    }
    cout<< edges_list.size() <<endl;

    // Decode the nodes in the updated list
    vector<pair<int, int> > decoded_list = decodeNodes(edges_list);

    // Write the decoded list to an output file
    ofstream output_file;
    output_file.open("decoded_edges.csv");

    for (int i = 0; i < decoded_list.size(); i++)
    {
        output_file << decoded_list[i].first << "," << decoded_list[i].second << endl;
    }

    cout << "Decoded edges successfully written to decoded_edges.csv" << endl;

    input_file.close();
    output_file.close();

    return 0;
}