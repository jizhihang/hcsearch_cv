#include "LossFunction.hpp"
#include "MyLogger.hpp"
#include "Globals.hpp"

namespace HCSearch
{
	/**************** Loss Functions ****************/

	HammingLoss::HammingLoss()
	{
	}

	HammingLoss::~HammingLoss()
	{
	}

	double HammingLoss::computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)
	{
		Matrix<bool, Dynamic, 1> diff = YPred.graph.nodesData.array() != YTruth.graph.nodesData.array();
		double loss = 0.0;
		for (int i = 0; i < diff.size(); i++)
		{
			if (diff(i))
				loss++;
		}
		return loss/diff.size();
	}

	PixelHammingLoss::PixelHammingLoss()
	{
	}

	PixelHammingLoss::~PixelHammingLoss()
	{
	}

	double PixelHammingLoss::computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)
	{
		if (!YTruth.nodeWeightsAvailable)
		{
			LOG(WARNING) << "node weights are not available for computing pixel hamming loss.";
		}

		Matrix<bool, Dynamic, 1> diff = YPred.graph.nodesData.array() != YTruth.graph.nodesData.array();
		double loss = 0.0;
		for (int i = 0; i < diff.size(); i++)
		{
			if (diff(i))
				if (YTruth.nodeWeightsAvailable)
					loss += YTruth.nodeWeights(i);
				else
					loss += 1.0/diff.size();
		}
		return loss;
	}

	DepthLoss::DepthLoss()
	{
	}

	DepthLoss::~DepthLoss()
	{
	}

	double DepthLoss::computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)
	{
		if (!YTruth.nodeWeightsAvailable)
		{
			LOG(WARNING) << "node weights are not available for computing depth loss.";
		}
		if (!YTruth.nodeDepthsAvailable || !Global::settings->CLASSES.isDepthsAvailable())
		{
			LOG(WARNING) << "node depths are not available for computing depth loss.";
		}

		double loss = 0.0;
		int numNodes = YTruth.nodeDepths.size();
		for (int i = 0; i < numNodes; i++)
		{
			int label = YPred.getLabel(i);
			double center = Global::settings->CLASSES.getDepth(label);

			double depthError = abs(log10(center) - log10(YTruth.nodeDepths(i)));
			//double relativeDepthError = abs(center - YTruth.nodeDepths(i))/YTruth.nodeDepths(i);
			loss += YTruth.nodeWeights(i)*depthError;
		}

		return loss;
	}
}