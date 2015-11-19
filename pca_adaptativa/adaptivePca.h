#ifndef ADAPTIVE_PCA
#define ADAPTIVE_PCA
#include <vector>

class AdaptivePCA {
	public:
	AdaptivePCA(int inputSize, int outputSize);
	~AdaptivePCA();
	int train(std::vector< double* > inputData, int epochs, double maxSideWeight);
	void evaluate(double* sample, double* output);

	private:
	void print();
	double calcMean(std::vector< double* > inputData, int index);
	double calcVariance(std::vector< double* > inputData, int index);
	void calcYVector(double* inputData, double* y);
	double trainSample(double* inputData, double learningRate, double sideLearningRate, double momentum);
	void normalizeWeights();
	void normalizeInputs(std::vector< double* > &inputData);

	int m_inputSize;
	int m_outputSize;
	double *m_weights; //w matrix
	double *m_sideWeights; // u matrix
};

#endif
