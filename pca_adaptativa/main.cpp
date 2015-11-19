#include "adaptivePca.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <list>
#include <cstdlib>
#include <algorithm>

#define EPOCHS 300

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

int main(int argc, char** argv) {
	std::ifstream file("iris.data");
	std::string line;
	int count = 0;
	std::vector<float*> inputList;
	std::vector<int> results;
	float normalizingFactors[4] = {0.0}; //keeps the max value of each attribute

	while (file.good()) {
		float *inputs = new float[4]();
		int result;

		getline(file, line);
		std::vector<std::string> tokens = split(line, ',');
		if (tokens.size() != 5) {
//			std::cout << "Tokens: " << tokens.size() << std::endl;
			continue;
		}

//		std::cout << "Line " << count << std::endl;
		for (int i=0;i<4;i++) {
			inputs[i] = atof(tokens[i].c_str());
			if (inputs[i] > normalizingFactors[i]) {
				normalizingFactors[i] = inputs[i];
			}
		}

		if (tokens[4].compare("Iris-setosa") == 0) {
			results.push_back(0);
		}
		else if (tokens[4].compare("Iris-versicolor") == 0) {
			results.push_back(1);
		}
		else {
			results.push_back(2);
		}
		inputList.push_back(inputs);

//		std::cout << "0: " << inputs[0] << ", ";
//		std::cout << "1: " << inputs[1] << ", ";
//		std::cout << "2: " << inputs[2] << ", ";
//		std::cout << "3: " << inputs[3] << ", ";
//		std::cout << "R: " << results.back() << std::endl << std::endl;

		count++;
	}
	
	AdaptivePCA pca(4, 4);
	pca.train(inputList, EPOCHS, 0.0001);

	std::cout << "processing finished in " << count << " cycles" << std::endl;
	return 0;
}
