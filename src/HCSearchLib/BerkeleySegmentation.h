#ifndef BERKELEYSEGMENTATION_HPP
#define BERKELEYSEGMENTATION_HPP

#include "DataStructures.hpp"
#include "MyPrimitives.hpp"

using namespace Eigen;
using namespace std;

namespace HCSearch
{
	typedef int Node_t;
	typedef MyPrimitives::Pair<Node_t, Node_t> Edge_t;

	/*!
	 * The %BerkeleySegmentationUCM represents the Berkeley segmentation as an Ultrametric Contour Map.
	 * The dual representation to the UCM is the %BerkeleySegmentationTree.
	 */
	class BerkeleySegmentationUCM // TODO TEST
	{
	private:
		ImgLabeling graph; //!< graph with superpixel nodes and edges connecting neighbors
		map< Edge_t, double > edgeWeights; //!< UCM value assignment to every graph edge
		vector<double> weightsList; //!< all unique UCM edge weights sorted from lowest to highest

	public:
		BerkeleySegmentationUCM();
		BerkeleySegmentationUCM(ImgLabeling graph, map< Edge_t, double > edgeWeights);
		~BerkeleySegmentationUCM();

		vector<double> getAllWeights();

		ImgLabeling getThresholdedGraph(double threshold);

		ImgLabeling getGraph();
		map< Edge_t, double > getEdgeWeights();
	};

	class BerkeleySegmentationUtilities
	{
	public:
		static vector<double> getAllUniqueUCMValues(map< Edge_t, double > edgeWeights);
	};
}

#endif