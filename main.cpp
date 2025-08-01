#include"activations.h"
#include <iostream>
#include <memory>
#include "model.h"
#include "dense.h"


using namespace std;

int main() {
    Model<float> model;
    
   
    model.add(make_unique<Dense<float>>(32,Activations<float>::relu));
    model.add(make_unique<Dense<float>>(10,Activations<float>::relu));  

    vector<float> input = {0.5, 1.2, 0.8, 0.3};
    auto output = model.forward(input);

    cout << "Model output: ";
    for (float val : output) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}