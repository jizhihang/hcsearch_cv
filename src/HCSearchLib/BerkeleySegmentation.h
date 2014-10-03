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

	class BSTNode;

	/*!
	 * The %BerkeleySegmentationTree represents the Berkeley segmentation as a hierarchical tree.
	 * The dual representation to the tree is the %BerkeleySegmentationUCM (Ultrametric Contour Map).
	 * However, the tree representation allows for more local operations on regions.
	 */
	class BerkeleySegmentationTree
	{
	private:
		enum BSTAction { STAY=0, SPLIT, MERGE, NO_ACTION };

		BSTNode* root;
		map<double, BSTNode*> levels; //!< all unique UCM edge weights represented as levels in tree
		vector<double> weightsList; //!< all unique UCM edge weights sorted from lowest to highest

		set<BSTNode*> currentPartition; //!< the current partitioning of regions

		BSTAction prevAction; //!< previous action taken
		BSTNode* prevActionedNode; //!< previous node that took action

	public:
		BerkeleySegmentationTree();
		
		// Must create on heap, or declare like so: BerkeleySegmentation bs(graph, edgeWeights);
		// since copy constructor is not implemented
		BerkeleySegmentationTree(ImgLabeling graph, map< Edge_t, double > edgeWeights);

		~BerkeleySegmentationTree();

		/*!
		 * Split the current region into two.
		 */
		bool splitRegion(BSTNode* node);

		/*!
		 * Merge the current region with its sibling into the parent.
		 */
		bool mergeRegion(BSTNode* node);

		/*!
		 * Undo the previous action.
		 */
		bool undoPrevAction();

		/*!
		 * Check if the region can be split.
		 */
		bool canSplitRegion(BSTNode* node);

		/*!
		 * Check if the region's sibling is part of the current partition.
		 */
		bool canMergeRegion(BSTNode* node);

		/*!
		 * Get the current partition.
		 */
		set<BSTNode*> getCurrentPartition();

		/*!
		 * Get the root node.
		 */
		BSTNode* getRoot();

	private:
		/*!
		 * Helper to construct the initial tree.
		 */
		void constructTreeHelper(ImgLabeling graph, 
			map< Edge_t, double > edgeWeights, vector<double> weightsList);

		/*!
		 * Delete the tree and free up memory.
		 */
		static void deleteTree(BSTNode* node);
	};

	/*!
	 * @brief %BSTNode is a node in the %BerkeleySegmentationTree.
	 * Leaf nodes represent superpixels and intermediate nodes store the UCM value between the children.
	 * Intermediate nodes have two interpretations:
	 *		1) the boundary (with a UCM value) that splits the children regions
	 *		2) the region that includes all the descendent superpixels
	 */
	class BSTNode
	{
	public:
		BSTNode* parent; //!< parent node, NULL if none
		BSTNode* childL; //!< left child node, NULL if none
		BSTNode* childR; //!< right child node, NULL if none
		int nodeID; //!< if leaf node, ID of superpixel; otherwise a unique ID for the intermediate node
		double ucmValue; //!< UCM value of edge between the two children nodes
		set<Node_t> descendentSuperpixels; //!< store descendent superpixel IDs for fast access

		bool isActivated; //!< if activated, then region is covered

	public:
		BSTNode();
		~BSTNode();

		set<Node_t> getAllDescendentSuperpixels();
		
		bool isLeafNode();
		bool isIntermediateNode();
		bool isRootNode();
	};
}

#endif