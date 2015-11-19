#include "adaptivePca.h"
#include <stdlib.h>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

AdaptivePCA::AdaptivePCA(int inputSize, int outputSize) {
	m_weights = new double[inputSize * outputSize]();
	m_sideWeights = new double[outputSize * outputSize]();

	m_inputSize = inputSize;
	m_outputSize = outputSize;
}

AdaptivePCA::~AdaptivePCA() {
	delete[] m_weights;
	delete[] m_sideWeights;
}

void AdaptivePCA::evaluate(double* sample, double* output) {
	for (int j=0;j<m_outputSize;j++) {
		output[j] = 0.0;

		//weights
		for (int i=0;i<m_inputSize;i++) {
			output[j] += m_weights[i * m_outputSize + j] * sample[i];
		}
	}
}

void AdaptivePCA::calcYVector(double* inputData, double* y) {
	for (int j=0;j<m_outputSize;j++) {
		y[j] = 0.0;

		//weights
		for (int i=0;i<m_inputSize;i++) {
			y[j] += m_weights[i * m_outputSize + j] * inputData[i];
		}

		//side weights
		for (int l=0;l<j;l++) {
			y[j] += m_sideWeights[l * m_outputSize + j] * y[l];
		}
	
	}
}

void AdaptivePCA::normalizeWeights() {
	for (int j=0;j<m_outputSize;j++) {
		double mag = 0.0f;

		for (int i=0;i<m_inputSize;i++) {
			mag += m_weights[i * m_outputSize + j] * m_weights[i * m_outputSize + j];
		}

		mag = sqrt(mag);

		for (int i=0;i<m_inputSize;i++) {
			m_weights[i * m_outputSize + j] /= mag;
		}
	}
}

void AdaptivePCA::normalizeInputs(std::vector< double* > &inputData) {
	for(int i=0;i<m_inputSize;i++) {
		double mean = calcMean(inputData, i);
		double variance = calcVariance(inputData, i);

		for (int j=0;j<inputData.size();j++) {
			inputData[j][i] = (inputData[j][i] - mean) / variance;
		}
	} 
}

/** 
* @brief train a single sample and adjust internal weights
* 
* @param inputData
* @param learningRate
* @param sideLearningRate
* @param momentum
* 
* @return value of the highest side weight
*/
double AdaptivePCA::trainSample(double* inputData, double learningRate, double sideLearningRate, double momentum) {
	double highestSideWeight = 0.0f;
	//calculate vector y
	double y[m_outputSize];
	calcYVector(inputData,y);

	//adjust weights
	for (int i=0;i<m_inputSize;i++) {
		for (int j=0;j<m_outputSize;j++) {
			//TODO introduce momentum terms
			double deltaW = learningRate * inputData[i] * y[j];
			m_weights[i * m_outputSize + j] += deltaW;
		}
	}

	normalizeWeights();	
	
	//adjust side weights
	for (int j=0;j<m_outputSize;j++) {
		for (int l=0;l<j;l++) {
			//TODO introduce momentum terms
			double deltaU = -sideLearningRate * y[l] * y[j];
			m_sideWeights[l * m_outputSize + j] += deltaU;
			if (fabs(m_sideWeights[l * m_outputSize + j]) > highestSideWeight) {
				highestSideWeight = fabs(m_sideWeights[l * m_outputSize + j]);
			}
		}
	}

	return highestSideWeight;
}

int AdaptivePCA::train(std::vector< double* > inputData, int epochs, double maxSideWeight) {
	int currentEpoch = 0;
	double highestSideWeight = 99999.9;
	double highestSideWeightFromEpoch = 99999.9;
	normalizeInputs(inputData);
	
	//init weights
	for (int i=0;i<m_inputSize * m_outputSize;i++) {
		m_weights[i] = (double)rand() / RAND_MAX;
	}

	for (int i=0;i<m_outputSize * m_outputSize;i++) {
		m_sideWeights[i] = (double)rand() / RAND_MAX;
	}

	while (currentEpoch <= epochs && highestSideWeightFromEpoch > maxSideWeight) {
		currentEpoch++;
		highestSideWeightFromEpoch = 0.0;
		double learningRate = 0.5 / currentEpoch;
		double sideLearningRate = 0.5 / currentEpoch;
		double momentum = 0.0;

//		std::cout << std::endl << "===========================================================" << std::endl;
//		std::cout << "Starting epoch " << currentEpoch << ". LearningRate: " << learningRate << ", SideLearningRate: " << sideLearningRate << std::endl;
		//randomize samples
		std::random_shuffle(inputData.begin(), inputData.end());
		
		for (int i=0;i<inputData.size();i++) {
			//TODO calculate learning rates and momentum

			highestSideWeight = trainSample(inputData[i], learningRate, sideLearningRate, momentum);
			if (highestSideWeightFromEpoch < highestSideWeight) {
				highestSideWeightFromEpoch = highestSideWeight;
			}

//			print();
		}

//		print();
//		std::cout << "Highest side weight: " << std::fixed << std::setprecision(10) << highestSideWeightFromEpoch << std::endl;
	}
	std::cout  << "Epochs: " << currentEpoch << std::endl;
	print();
}

double AdaptivePCA::calcMean(std::vector< double* > inputData, int index) {
	double sum = 0.0f;

	for (int i=0;i<inputData.size();i++) {
		sum += inputData[i][index];
	}

	return sum / inputData.size();
}

double AdaptivePCA::calcVariance(std::vector< double* > inputData, int index) {
	double mean = calcMean(inputData, index);
	double sum = 0.0f;

	for (int i=0;i<inputData.size();i++) {
		sum += pow(inputData[i][index] - mean, 2);
	}

	return sum / inputData.size();
}

void AdaptivePCA::print() {

	std::cout << "W: " << std::endl;
	for (int i=0;i<m_inputSize;i++) {
		for (int j=0;j<m_outputSize;j++) {
			std::cout << m_weights[i*m_outputSize + j] << "\t";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	std::cout << "U: " << std::endl;
	for (int j=0;j<m_outputSize;j++) {
		for (int l=0;l<j;l++) {
			std::cout << m_sideWeights[l*m_outputSize + j] << "\t";
		}
		std::cout << std::endl;
	}
		std::cout << std::endl;
}
