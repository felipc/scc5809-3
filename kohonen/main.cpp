#include "kohonen.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <list>
#include <cstdlib>
#include <algorithm>

#define WIDTH 25
#define HEIGHT 25
#define EPOCHS 300

#define DIM 13

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
	float normalizingFactors[DIM] = {0.0}; //keeps the max value of each attribute

	while (file.good()) {
		float *inputs = new float[DIM]();
		int result;

		getline(file, line);
		std::vector<std::string> tokens = split(line, ',');
		if (tokens.size() != DIM + 1) { // all inputs + the result at the first token
			std::cout << "Tokens: " << tokens.size() << std::endl;
			continue;
		}

		results.push_back(atof(tokens[0].c_str()));
//		std::cout << "Line " << count << std::endl;

		for (int i=0;i<DIM;i++) {
			inputs[i] = atof(tokens[i + 1].c_str());
			if (inputs[i] > normalizingFactors[i]) {
				normalizingFactors[i] = inputs[i];
			}
		}

		inputList.push_back(inputs);

		// std::cout << "0: " << inputs[0] << ", ";
		// std::cout << "1: " << inputs[1] << ", ";
		// std::cout << "2: " << inputs[2] << ", ";
		// std::cout << "3: " << inputs[3] << ", ";
		// std::cout << "R: " << results.back() << std::endl << std::endl;

		count++;
	}

	//normalize input data
	for (int i=0;i<inputList.size();i++) {
		for (int j=0;j<DIM;j++) {
			inputList[i][j] /= normalizingFactors[j];
		}
		// std::cout << "Line: " << i << ", ";
		// std::cout << "0: " << inputList[i][0] << ", ";
		// std::cout << "1: " << inputList[i][1] << ", ";
		// std::cout << "2: " << inputList[i][2] << ", ";
		// std::cout << "3: " << inputList[i][3] << ", ";
		// std::cout << "R: " << results[i] << std::endl << std::endl;

	}


	std::vector<float*> originalInputList(inputList);
	KohonenNetwork kohonen(WIDTH, HEIGHT, DIM, EPOCHS);

	//test functions
	for (int i=0;i<HEIGHT;i++) {
		for (int j=0;j<WIDTH;j++) {
			for (int k=0;k<DIM;k++) {
				int index = (i * WIDTH * DIM) + (j * DIM) + k;
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
		std::random_shuffle(inputList.begin(), inputList.end());
		for (int i=0;i<inputList.size();i++) {
			float change = kohonen.train(inputList[i]);
		}
	}
		
	//evaluate all results
	int histograms[3][WIDTH * HEIGHT] = {0};

printf("aa\n");
	for (int i=0;i<originalInputList.size();i++) {
		int index = kohonen.evaluate(originalInputList[i]);
		printf("index: %d\n", index);
		histograms[results[i]][index / 3]++;
	};

	printf("nn\n");

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
