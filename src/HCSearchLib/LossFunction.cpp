#include "LossFunction.hpp"
#include "MyLogger.hpp"

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
		if (!YTruth.nodeDepthsAvailable)
		{
			LOG(WARNING) << "node depths are not available for computing depth loss.";
		}

		// TODO: remove hard code
		vector<double> centers;
		centers.push_back(3.3293);
		centers.push_back(4.6733);
		centers.push_back(6.1544);
		centers.push_back(7.9779);
		centers.push_back(9.8224);
		centers.push_back(11.580);
		centers.push_back(13.503);
		centers.push_back(15.593);
		centers.push_back(18.064);
		centers.push_back(20.734);
		centers.push_back(23.624);
		centers.push_back(27.008);
		centers.push_back(30.882);
		centers.push_back(35.421);
		centers.push_back(40.498);
		centers.push_back(47.353);
		centers.push_back(55.278);
		centers.push_back(65.359);
		centers.push_back(79.974);
		centers.push_back(81.914);

		double loss = 0.0;
		int numNodes = YTruth.nodeDepths.size();
		for (int i = 0; i < numNodes; i++)
		{
			int label = YPred.getLabel(i);
			double center = centers[label-1];

			double depthError = abs(log10(center) - log10(YTruth.nodeDepths(i)));
			loss += YTruth.nodeWeights(i)*depthError;
		}

		return loss;
	}
}