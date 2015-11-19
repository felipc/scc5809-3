#include "adaptivePca.h"
#include <stdlib.h>
#include <cmath>

AdaptivePCA::AdaptivePCA(int inputSize, int outputSize) {
	m_weights = new float[inputSize * outputSize]();
	m_sideWeights = new float[outputSize * outputSize]();

	m_inputSize = inputSize;
	m_outputSize = outputSize;
}

AdaptivePCA::~AdaptivePCA() {
	delete[] m_weights;
	delete[] m_sideWeights;
}

void AdaptivePCA::calcYVector(float* inputData, float* y) {
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

//return max side weight
float AdaptivePCA::trainSample(float* inputData, float learningRate, float sideLearningRate, float momentum) {
	//calculate vector y
	float y[m_outputSize];

	calcYVector(inputData,y);

	//adjust weights
	for (int i=0;i<m_inputSize;i++) {
		for (int j=0;j<m_outputSize;j++) {
			//TODO introduce momentum terms
			float deltaW = learningRate * inputData[i] * y[i];
			m_weights[i * m_outputSize + j] += deltaW;
		}
	}

	//adjust side weights
	for (int j=0;j<m_outputSize;j++) {
		for (int l=0;l<j;l++) {
			//TODO introduce momentum terms
			float deltaU = -sideLearningRate * y[l] * y[j];
			m_sideWeights[l * m_outputSize + j] += deltaU;
		}
	}
}

int AdaptivePCA::train(std::vector< float* > inputData, int epochs, float maxSideWeight) {
	int currentEpoch = 0;
	float sideWeight = 0.0f;

	//TODO normalize data
	
	//init weights
	for (int i=0;i<m_inputSize * m_outputSize;i++) {
		m_weights[i] = (float)rand() / RAND_MAX;
	}

	while (currentEpoch < epochs && sideWeight > maxSideWeight) {
		//randomize samples
		std::random_shuffle(inputData.begin(), inputData.end());
		

	}
}

float AdaptivePCA::calcMean(std::vector< float* > inputData, int index) {
	float sum = 0.0f;

	for (int i=0;i<inputData.size();i++) {
		sum += *inputData[index];
	}

	return sum / inputData.size();
}

float AdaptivePCA::calcVariance(std::vector< float* > inputData, int index) {
	float mean = calcMean(inputData, index);
	float sum = 0.0f;

	for (int i=0;i<inputData.size();i++) {
		sum += pow(*inputData[index] - mean, 2);
	}

	return sum / inputData.size();
}

