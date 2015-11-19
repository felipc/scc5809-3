#ifndef KOHONEN_H
#define KOHONEN_H

class KohonenNetwork {
	public:
	KohonenNetwork(int sizeX, int sizeY, int inputs, int trainingIterations);
	float train(const float* inputs);
	int evaluate(const float* inputs);
	float distance(const float* a, const float* b);
	int x(int index);
	int y(int index);
	float distanceToNode(int indexA, int indexB);

//	private:

	int m_sizeX;
	int m_sizeY;
	int m_inputs;
	int m_trainingIterations;
	int m_currentIteration;
	int m_currentEpoch;
	float* m_weights;

	float m_mapRadius;
	float m_timeConstant;
	float m_startLearningRate;

};

#endif
