#include "adaptivePca.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <list>
#include <cstdlib>
#include <algorithm>

#define EPOCHS 300000

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
	std::vector<double*> inputList;
	std::vector<double*> originalInputList;
	std::vector<int> results;

	while (file.good()) {
		double *inputs = new double[4]();
		double *originalInputs = new double[4]();

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
			originalInputs[i] = inputs[i];
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
		originalInputList.push_back(originalInputs);
		count++;
	}
	
	AdaptivePCA pca(4, 2);
	pca.train(inputList, EPOCHS, 0.0001);

	double output[2];

	for (int i=0;i<originalInputList.size();i++) {
//		std::cout << "sample " << i << ": ";
//		for (int j=0;j<4;j++) {
//			std::cout << originalInputList[i][j] << ", ";
//		}
		pca.evaluate(originalInputList[i], output);
		std::cout << output[0] << "," << output[1] << "," << results[i] << std::endl;

//		std::cout << std::endl;
	}

	return 0;
}
