#ifndef ADAPTIVE_PCA
#define ADAPTIVE_PCA
#include <vector>

class AdaptivePCA {
	public:
	AdaptivePCA(int inputSize, int outputSize);
	~AdaptivePCA();
	int train(std::vector< float* > inputData, int epochs, float maxSideWeight);

	private:
	float calcMean(std::vector< float* > inputData, int index);
	float calcVariance(std::vector< float* > inputData, int index);
	void calcYVector(float* inputData, float* y);
	float trainSample(float* inputData, float learningRate, float sideLearningRate, float momentum);

	int m_inputSize;
	int m_outputSize;
	float *m_weights; //w matrix
	float *m_sideWeights; // u matrix
};

#endif
