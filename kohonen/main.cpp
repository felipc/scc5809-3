#include "kohonen.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <list>
#include <cstdlib>
#include <algorithm>

#define WIDTH 15
#define HEIGHT 15
#define EPOCHS 100
#define ATTRIBUTES 13

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
	std::ifstream file("vinhos.txt");
	std::string line;
	int count = 0;
	std::vector<float*> inputList;
	std::vector<int> results;
	float normalizingFactors[ATTRIBUTES] = {0.0}; //keeps the max value of each attribute

	while (file.good()) {
		float *inputs = new float[ATTRIBUTES]();
		int result;

		getline(file, line);
		std::vector<std::string> tokens = split(line, ',');
		if (tokens.size() != ATTRIBUTES + 1) {
			std::cout << "Tokens: " << tokens.size() << std::endl;
			continue;
		}

//		std::cout << "Line " << count << std::endl;
		for (int i=0;i<ATTRIBUTES;i++) {
			inputs[i] = atof(tokens[i+1].c_str());
			if (inputs[i] > normalizingFactors[i]) {
				normalizingFactors[i] = inputs[i];
			}
			std::cout << i << ": " << inputs[i] << ", ";
		}
//		inputs[0] =  atof(tokens[0].c_str());
//		inputs[1] =  atof(tokens[1].c_str());
//		inputs[2] =  atof(tokens[2].c_str());
//		inputs[3] =  atof(tokens[3].c_str());
		results.push_back(atoi(tokens[0].c_str()) - 1);
		inputList.push_back(inputs);

//		std::cout << "0: " << inputs[0] << ", ";
//		std::cout << "1: " << inputs[1] << ", ";
//		std::cout << "2: " << inputs[2] << ", ";
//		std::cout << "3: " << inputs[3] << ", ";
		std::cout << "R: " << results.back() << std::endl << std::endl;

		count++;
	}

	//normalize input data
	for (int i=0;i<inputList.size();i++) {
		std::cout << "Sample " << i << " -> ";
		for (int j=0;j<ATTRIBUTES;j++) {
			inputList[i][j] /= normalizingFactors[j];
			std::cout << j << ": " << inputList[i][j] << ", ";
		}
		std::cout << std::endl;
//		std::cout << "Line: " << i << ", ";
//		std::cout << "0: " << inputList[i][0] << ", ";
//		std::cout << "1: " << inputList[i][1] << ", ";
//		std::cout << "2: " << inputList[i][2] << ", ";
//		std::cout << "3: " << inputList[i][3] << ", ";
//		std::cout << "R: " << results.back() << std::endl << std::endl;

	}

	std::vector<float*> originalInputList(inputList);
	KohonenNetwork kohonen(WIDTH, HEIGHT, ATTRIBUTES, EPOCHS);

	//test functions
	for (int i=0;i<HEIGHT;i++) {
		for (int j=0;j<WIDTH;j++) {
			for (int k=0;k<ATTRIBUTES;k++) {
				int index = (i * WIDTH * ATTRIBUTES) + (j * ATTRIBUTES) + k;
				int x = kohonen.x(index);
				int y = kohonen.y(index);

				if (x != j || y != i) {
					std::cout << "ERROR (" << index << "). (x, y) = " << x << ", " << y << ";";
					std::cout << " (j, i) = " << j << ", " << i << ". K = " << k << std::endl;
				}
			}
		}
	}

	count = 0;
	while (count < EPOCHS) {
		count++;
//		std::random_shuffle(inputList.begin(), inputList.end());
		for (int i=0;i<inputList.size();i++) {
			float change = kohonen.train(inputList[i]);
		}
	}
		
	//evaluate all results
	int histograms[3][WIDTH * HEIGHT] = {0};

	for (int i=0;i<originalInputList.size();i++) {
		std::cout << "Evaluating sample " << i << std::endl;
		int index = kohonen.evaluate(originalInputList[i]);
		histograms[results[i]][index / ATTRIBUTES]++;
	};

	//generate image
	std::ofstream outFile("kohonen.ppm");
	outFile << "P3 " << WIDTH << " " << HEIGHT << " 255" << std::endl;
	for (int i=0;i<HEIGHT;i++) {
		for (int j=0;j<WIDTH;j++) {
			int index = (HEIGHT * i) + j;
			int pixelCount = histograms[0][index] + histograms[1][index] + histograms[2][index];

			if (j > 0) {
				outFile << " ";
			}

			if (pixelCount == 0) {
				pixelCount = 1; //avoid division by 0
				outFile << "255 255 255";
			}
			else {
				outFile << histograms[0][index] * 255 / pixelCount << " " << histograms[1][index] * 255 / pixelCount << " " << histograms[2][index] * 255 / pixelCount;
			}
		}
		outFile << std::endl;
	}



	std::cout << "processing finished in " << count << " cycles" << std::endl;
	return 0;
}
