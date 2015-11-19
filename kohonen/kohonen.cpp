#include "kohonen.h"
#include <cstdlib>
#include <float.h>
#include <cmath>
#include <vector>
#include <iostream>

KohonenNetwork::KohonenNetwork(int sizeX, int sizeY, int inputs, int trainingIterations) {
	m_weights = new float[sizeX * sizeY * inputs]();
	m_sizeX = sizeX;
	m_sizeY = sizeY;
	m_inputs = inputs;
	m_trainingIterations = trainingIterations;
	m_currentIteration = 0;
	m_currentEpoch = 0;

	m_mapRadius = std::max(sizeX, sizeY) / 2.0;
	m_timeConstant = m_trainingIterations / log(m_mapRadius);
	m_startLearningRate = 0.1;

	for (int i=0;i<sizeX * sizeY * inputs;i++) {
		m_weights[i] = (float)rand() / RAND_MAX;
	}
}

int KohonenNetwork::x(int index) {
	return index / m_inputs % m_sizeY;
}

int KohonenNetwork::y(int index) {
	return index / (m_inputs * m_sizeX);
}

float KohonenNetwork::distanceToNode(int indexA, int indexB) {
	int xA, xB, yA, yB;

	xA = x(indexA);
	xB = x(indexB);
	yA = y(indexA);
	yB = y(indexB);
	
//	std::cout << "indexA: " << indexA << ", X: " << xA << ", Y: " << yA << std::endl;
//	std::cout << "indexB: " << indexA << ", X: " << xB << ", Y: " << yB << std::endl;
	
	float dist = sqrt(((xA - xB) * (xA - xB)) + ((yA - yB) * (yA - yB)));
//	std::cout << "Dist: " << dist << std::endl << std::endl;
	return dist;
}

float KohonenNetwork::train(const float* inputs) {
	int index = evaluate(inputs); //locate the activated neuron
	float maxDelta = 0.0;
	float neighborhoodRadius = m_mapRadius  * exp((double) -m_currentEpoch / m_timeConstant);
	float learningRate = m_startLearningRate * exp((double)-m_currentEpoch / m_trainingIterations);
	
	std::cout << "Iteration: " << m_currentEpoch << ". LearningRate: " << learningRate << ". Neighborhood: " << neighborhoodRadius << std::endl;

	for(int j=0;j<m_sizeX * m_sizeY * m_inputs;j+=m_inputs) {
		float dist = distanceToNode(j, index);
		float theta = exp(-(dist * dist) / (2.0 * neighborhoodRadius*neighborhoodRadius));
//		std::cout << "Dist: " << dist << ", Theta: " << theta << std::endl;
		if (dist * dist > neighborhoodRadius) { 
			continue;
		}

		for (int i=0;i<4;i++) {
			float delta = theta * learningRate  * (inputs[i] - m_weights[j+i]);
			m_weights[j+i] += delta;
			if (delta > maxDelta) {
				maxDelta = delta;
			}
		}
	}

	m_currentIteration++;
	if (m_currentIteration % 150 == 0) {
		m_currentEpoch++;
	}
	return maxDelta;
}

int KohonenNetwork::evaluate(const float* inputs) {
	int minId = 0;
	float minValue = FLT_MAX;

	for (int i=0;i<m_sizeX * m_sizeY * m_inputs;i+=m_inputs) {
		float v = distance(m_weights + i, inputs);
		if (v < minValue) {
			minId = i;
			minValue = v;
		}
	}
	return minId;
}

float KohonenNetwork::distance(const float* a, const float* b) {
	float d = 0.0;

	for (int i=0;i<m_inputs;i++) {
		d += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(d);
}
