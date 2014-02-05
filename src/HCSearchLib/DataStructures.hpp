#ifndef DATASTRUCTURES_HPP
#define DATASTRUCTURES_HPP

#include <map>
#include <set>
#include <fstream>
#include "../../external/Eigen/Eigen/Dense"
#include "MyPrimitives.hpp"
#include "Settings.hpp"

using namespace Eigen;
using namespace std;

namespace HCSearch
{
	/**************** Constants ****************/

	enum CompareSearchNodeType { HEURISTIC, COST };
	enum SearchType { LL=0, HL, LC, HC, 
		LEARN_H, LEARN_C, LEARN_C_ORACLE_H };
	enum DatasetType { TEST=0, TRAIN, VALIDATION };

	const extern string SearchTypeStrings[];
	const extern string DatasetTypeStrings[];

	/**************** Graph ****************/

	/*!
	 * Set of (neighbor) nodes.
	 */
	typedef set< int > NeighborSet_t;

	/*!
	 * Adjacency List. 
	 * Mapping: node -> set of neighbor nodes
	 */
	typedef map< int, NeighborSet_t > AdjList_t;

	/*!
	 * @brief Basic graph implementation with feature data at nodes.
	 * 
	 * This data structure is publicly accessible so you can perform 
	 * direct reading and manipulations.
	 */
	struct FeatureGraph
	{
		/*!
		 * Node data stores features at nodes.
		 * Rows = nodes, cols = feature data
		 */
		MatrixXd nodesData;

		/*!
		 * Adjacency list of the graph.
		 * Node -> set of neighbor nodes
		 */
		AdjList_t adjList;
	};

	/*!
	 * @brief Basic graph implementation with label data at nodes. 
	 * 
	 * This data structure is publicly accessible so you can perform 
	 * direct reading and manipulations.
	 */
	struct LabelGraph
	{
		/*!
		 * Nodes with labels.
		 */
		VectorXi nodesData;

		/*!
		 * Adjacency list of the graph.
		 * Node -> set of neighbor nodes
		 */
		AdjList_t adjList;
	};

	/**************** Features and Labelings ****************/

	/*!
	 * @brief Structured input: graph with feature nodes.
	 * 
	 * This stores a graph where nodes hold feature data. 
	 * Usually the nodes correspond to pixels, superpixels or patches, 
	 * and the data at each node correspond to the features of that
	 * pixel, superpixel or patch.
	 */
	class ImgFeatures
	{
	public:
		/*!
		 * Graph with feature nodes. Public access allows "low level" manipulations.
		 */
		FeatureGraph graph;

		/*!
		 * File name of this image.
		 */
		string filename;

	public:
		ImgFeatures();
		~ImgFeatures();

		/*!
		 * Convenience function to get the feature dimension of nodes.
		 * @return Returns the feature dimension
		 */
		int getFeatureDim();

		/*!
		 * Convenience function to get the number of nodes.
		 * @return Returns the number of nodes
		 */
		int getNumNodes();

		/*!
		 * Convenience function to get a feature component at a node.
		 * @param[in] node Node index
		 * @param[in] featIndex Index of feature component
		 * @return Returns the feature component of the node
		 */
		double getFeature(int node, int featIndex);

		/*!
		 * Get the file name that the features came from.
		 * @return Returns the file name
		 */
		string getFileName();
	};

	/*!
	 * @brief Structured output labeling: graph with label nodes.
	 * 
	 * This stores a graph where nodes hold label data. 
	 * Usually the nodes correspond to pixels, superpixels or patches. 
	 * Each pixel, superpixel or patch has an integer label usually corresponding 
	 * to a class.
	 */
	class ImgLabeling
	{
	public:
		/*!
		 * Graph with label nodes. Public access allows "low level" manipulations.
		 */
		LabelGraph graph;

		/*!
		 * Class confidences of nodes from initial prediction. 
		 * rows = nodes, cols = label probabilities (sorted by class index)
		 * Make sure to check if they are available using ImgLabeling::confidencesAvailable.
		 */
		MatrixXd confidences;

		/*!
		 * Adjacency list of edges that survived stochastic cuts.  
		 * node -> set of neighbor nodes
		 * Make sure to check if they are available using ImgLabeling::stochasticCutsAvailable.
		 */
		map< int, set<int> > stochasticCuts;

		bool confidencesAvailable;
		bool stochasticCutsAvailable;

	public:
		ImgLabeling();
		~ImgLabeling();

		/*!
		 * Convenience function to get the number of nodes.
		 * @return Returns the number of nodes
		 */
		int getNumNodes();

		/*!
		 * Convenience function to get a node's label.
		 * @param[in] node Node index
		 * @return Returns the label of the node
		 */
		int getLabel(int node);

		/*!
		 * Convenience function to get labels of the node's neighbors.
		 * @param[in] node Node index
		 * @return Returns the set of labels of neighboring nodes
		 */
		set<int> getNeighborLabels(int node);

		/*!
		 * Convenience function to get node's neighbors.
		 * @param[in] node Node index
		 * @return Returns the set of neighboring nodes (indices)
		 */
		set<int> getNeighbors(int node);

		/*!
		 * Convenience function to check if node has neighbors.
		 * @param[in] node Node index
		 * @return Returns true if there are neighbors
		 */
		bool hasNeighbors(int node);
	};

	/**************** Rank Features ****************/

	/*!
	 * @brief Stores features for ranking.
	 * 
	 * This is nothing more than a wrapper around a VectorXd object.
	 */
	class RankFeatures
	{
	public:
		/*!
		 * Features data
		 */
		VectorXd data;

		/*!
		 * Default constructor does nothing.
		 */
		RankFeatures();

		/*!
		 * Constructor to initialize features data.
		 */
		RankFeatures(VectorXd features);
		
		~RankFeatures();
	};

	/**************** Rank Model ****************/

	/*!
	 * @brief Abstract class for model/weights for ranking.
	 * 
	 * The main purpose is to rank features using a rank model. 
	 * Abstract class is useful for extending different kinds of 
	 * rankers, like offline vs. online and linear vs. nonlinear.
	 */
	class IRankModel
	{
	protected:
		bool initialized;

	public:
		virtual ~IRankModel() {}

		/*!
		 * Use the model to rank the features.
		 * @param[in] features Features for ranking
		 * @return Returns the ranking of the feature
		 */
		virtual double rank(RankFeatures features)=0;

		/*!
		 * Get the ranker type.
		 */
		virtual RankerType rankerType()=0;

		/*!
		 * Load weights data from file.
		 */
		virtual void load(string fileName)=0;

		/*!
		 * Save weights data to file.
		 */
		virtual void save(string fileName)=0;

	protected:
		/*!
		 * Manual implementation of the vector dot product. 
		 * Used to correct a mysterious problem with Eigen's implementation.
		 */
		template <class TVector>
		static double vectorDot(TVector v1, TVector v2)
		{
			double result = 0;
			for (int i = 0; i < v1.size(); i++)
			{
				if (i >= v2.size())
					break;

				result += v1(i)*v2(i);
			}
			return result;
		}
	};

	/**************** SVM-Rank Model ****************/

	/*!
	 * @brief Rank model for SVM-Rank.
	 * 
	 * Has methods to learn weights from training examples and ranking.
	 */
	class SVMRankModel : public IRankModel
	{
	private:
		/*!
		 * SVM-rank weights
		 */
		VectorXd weights;

		/*!
		 * Output stream to training file for learning
		 */
		ofstream* rankingFile;

		/*!
		 * Training file name
		 */
		string rankingFileName;

		/*!
		 * Cumulative QID for learning
		 */
		int qid;

		/*!
		 * True if currently used for learning
		 */
		bool learningMode;

	public:
		SVMRankModel();

		/*!
		 * Construct with SVM weights from model file.
		 */
		SVMRankModel(string fileName);
		
		virtual double rank(RankFeatures features);
		virtual RankerType rankerType();
		virtual void load(string fileName);
		virtual void save(string fileName);

		/*!
		 * Get weights.
		 */
		VectorXd getWeights();

		/*!
		 * Initialize learning.
		 */
		void startTraining(string featuresFileName);

		/*!
		 * Add training examples.
		 */
		void addTrainingExamples(vector< RankFeatures >& betterSet, vector< RankFeatures >& worseSet);

		/*!
		 * End learning.
		 *
		 * Calls SVM Rank program to train on examples and produce model.
		 */
		void finishTraining(string modelFileName);

		/*!
		 * Cancel learning. Closes training file.
		 */
		void cancelTraining();

	private:
		/*!
		 * Load weights from file. 
		 * File format is the SVM-Rank model file 
		 * (weights are on the 12th line).
		 */
		static VectorXd parseModelFile(string fileName);

		/*!
		 * Convert vector into SVM-Rank line.
		 */
		static string vector2svmrank(RankFeatures features, int target, int qid);

		/*!
		 * Write weights to file.
		 * 
		 * File format:
		 *     ...
		 *	   line 12: 1:val 2:val ...
		 */
		static void writeModelFile(string fileName, const VectorXd& weights);
	};

	/**************** Online Rank Model ****************/

	/*!
	 * @brief Rank model for online passive-aggressive ranking. 
	 * 
	 * Has methods to perform online updates and ranking
	 */
	class OnlineRankModel : public IRankModel
	{
	private:
		/*!
		 * Latest weights.
		 */
		VectorXd latestWeights;

		/*!
		 * Cumulative sum of weights.
		 */
		VectorXd cumSumWeights;

		/*!
		 * Number of weights in cumulative sum for averaging. 
		 * -1 => OnlineRankModel not initialized
		 */
		int numSum;

	public:
		OnlineRankModel();

		/*!
		 * Construct with online weights from model file.
		 */
		OnlineRankModel(string fileName);

		virtual double rank(RankFeatures features);
		virtual RankerType rankerType();
		virtual void load(string fileName);
		virtual void save(string fileName);

		/*!
		 * Get latest weights.
		 */
		VectorXd getLatestWeights();

		/*!
		 * Get averaged weights.
		 */
		VectorXd getAvgWeights();

		/*!
		 * Perform an update. 
		 * - delta = loss (bad) - loss (good)
		 * - featureDiff = feature (good) - feature (bad)
		 */
		void performOnlineUpdate(double delta, VectorXd featureDiff);
		
		/*!
		 * Initialize weights to zero vector with dimension dim.
		 */
		void initialize(int dim);

	private:
		/*!
		 * Load weights from file.
		 * 
		 * File format:
		 *     line 1 (numsum): int
		 *     line 2 (cumsumweights): 1:val 2:val ...
		 *	   line 3 (latestweights): 1:val 2:val ...
		 */
		static void parseModelFile(string fileName, VectorXd& latestWeights, VectorXd& cumSumWeights, int& numSum);

		/*!
		 * Write weights to file.
		 * 
		 * File format:
		 *     line 1 (numsum): int
		 *     line 2 (cumsumweights): 1:val 2:val ...
		 *	   line 3 (latestweights): 1:val 2:val ...
		 */
		static void writeModelFile(string fileName, const VectorXd& latestWeights, const VectorXd& cumSumWeights, int numSum);
	};
}

#endif