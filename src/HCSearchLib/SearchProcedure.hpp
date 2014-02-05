#ifndef SEARCHPROCEDURE_HPP
#define SEARCHPROCEDURE_HPP

#include <vector>
#include "DataStructures.hpp"
#include "SearchSpace.hpp"

namespace HCSearch
{
	/**************** Save Prediction ****************/

	/*!
	 * @brief Convenience functions to save predictions.
	 */
	class SavePrediction
	{
	public:
		/*!
		 * Save the labels of the labeling.
		 */
		static void saveLabels(ImgLabeling& YPred, string fileName);

		/*!
		 * Save the stochastic cuts of the labeling.
		 */
		static void saveCuts(ImgLabeling& YPred, string fileName);
	};

	/*!
	 * @defgroup SearchProcedure Search Procedure
	 * @brief Provides an interface for setting up a search procedure.
	 * @{
	 */

	/**************** Search Procedure ****************/

	/*!
	 * @brief Search procedure abstract class. 
	 * Implement a search procedure by extending this class.
	 */
	class ISearchProcedure
	{
	public:
		// Meta data for an instance of search
		struct SearchMetadata
		{
			// save anytime results during search if true
			bool saveAnytimePredictions;

			// train, validation or testing
			DatasetType setType;

			// image example name
			string exampleName;

			// stochastic iteration
			int iter;

		public:
			SearchMetadata();
		};
	
	protected:
		class ISearchNode;
		class LLSearchNode;
		class HLSearchNode;
		class LCSearchNode;
		class HCSearchNode;
		class LearnHSearchNode;
		class LearnCSearchNode;
		class LearnCOracleHSearchNode;
		class CompareByHeuristic;
		class CompareByCost;

		typedef priority_queue<ISearchNode*, vector<ISearchNode*>, CompareByHeuristic> SearchNodeHeuristicPQ;
		typedef priority_queue<ISearchNode*, vector<ISearchNode*>, CompareByCost> SearchNodeCostPQ;

	public:
		virtual ~ISearchProcedure() {}

		/*!
		 * @brief Search procedure implemented by extending class.
		 * 
		 * Accepts features X and a model (and groudtruth Y if applicable) and performs search.
		 */
		virtual ImgLabeling searchProcedure(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
			int timeBound, SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel, SearchMetadata searchMetadata)=0;

		/*!
		 * @brief Convenience function for LL-search.
		 */
		ImgLabeling llSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, 
			SearchSpace* searchSpace, SearchMetadata searchMetadata);

		/*!
		 * @brief Convenience function for HL-search.
		 */
		ImgLabeling hlSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, 
			SearchSpace* searchSpace, IRankModel* heuristicModel, SearchMetadata searchMetadata);

		/*!
		 * @brief Convenience function for LC-search.
		 */
		ImgLabeling lcSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, 
			SearchSpace* searchSpace, IRankModel* costModel, SearchMetadata searchMetadata);

		/*!
		 * @brief Convenience function for HC-search.
		 */
		ImgLabeling hcSearch(ImgFeatures& X, int timeBound, SearchSpace* searchSpace, 
			IRankModel* heuristicModel, IRankModel* costModel, SearchMetadata searchMetadata);

		/*!
		 * @brief Convenience function for learning H search.
		 */
		void learnH(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
			IRankModel* learningModel, SearchMetadata searchMetadata);

		/*!
		 * @brief Convenience function for learning C search.
		 */
		void learnC(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
			IRankModel* heuristicModel, IRankModel* learningModel, SearchMetadata searchMetadata);

		/*!
		 * @brief Convenience function for learning C with oracle H search.
		 */
		void learnCWithOracleH(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
			IRankModel* learningModel, SearchMetadata searchMetadata);

	protected:
		void saveAnyTimePrediction(ImgLabeling YPred, int timeBound, SearchMetadata searchMetadata, SearchType searchType);
		void trainHeuristicRanker(IRankModel* ranker, SearchNodeHeuristicPQ& candidateSet, vector< ISearchNode* > successorSet);
		void trainCostRanker(IRankModel* ranker, SearchNodeCostPQ& costSet);
	};

	/*!
	 * @brief Basic search procedure abstract definition.
	 * Implements a generic search procedure, where you must define the virtual "stubs."
	 */
	class IBasicSearchProcedure : public ISearchProcedure
	{
	public:
		virtual ImgLabeling searchProcedure(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
			int timeBound, SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel, SearchMetadata searchMetadata);

	protected:
		/*!
		 * @brief Stub for selecting a subset of the open set for processing.
		 */
		virtual vector< ISearchNode* > selectSubsetOpenSet(SearchNodeHeuristicPQ& openSet)=0;

		/*!
		 * @brief Stub for expanding the elements.
		 * 
		 * openSet may be modified. costSet is used for duplicate checking.
		 */
		virtual SearchNodeHeuristicPQ expandElements(vector< ISearchNode* > subsetOpenSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet)=0;

		/*!
		 * @brief Stub for choosing successors among the expanded.
		 * 
		 * Returns the successors and adds the successors to the openSet and costSet.
		 * Side effect: candidate set has worst states remaining after function call.
		 */
		virtual vector< ISearchNode* > chooseSuccessors(SearchNodeHeuristicPQ& candidateSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet)=0;

		/*!
		 * @brief Checks if the state is duplicate among the states in the priority queue.
		 */
		template <class T>
		bool isDuplicate(ISearchNode* state, T& pq);

		/*!
		 * @brief Empty priority queue and delete all elements.
		 */
		template <class T>
		void deleteQueueElements(T& queue);
	};

	/**************** Beam Search Procedure ****************/

	/*!
	 * @brief Beam search procedure abstract definition.
	 */
	class IBeamSearchProcedure : public IBasicSearchProcedure
	{
	protected:
		static const int DEFAULT_BEAM_SIZE = 1;

		int beamSize; //!< Beam size
	};

	/**************** Breadth-First Beam Search Procedure ****************/

	/*!
	 * @brief Breadth-first beam search procedure.
	 */
	class BreadthFirstBeamSearchProcedure : public IBeamSearchProcedure
	{
	public:
		BreadthFirstBeamSearchProcedure();
		BreadthFirstBeamSearchProcedure(int beamSize);
		~BreadthFirstBeamSearchProcedure();

		virtual vector< ISearchNode* > selectSubsetOpenSet(SearchNodeHeuristicPQ& openSet);
		virtual SearchNodeHeuristicPQ expandElements(vector< ISearchNode* > subsetOpenSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet);
		virtual vector< ISearchNode* > chooseSuccessors(SearchNodeHeuristicPQ& candidateSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet);
	};

	/**************** Best-First Beam Search Procedure ****************/

	/*!
	 * @brief Best-first beam search procedure.
	 */
	class BestFirstBeamSearchProcedure : public BreadthFirstBeamSearchProcedure
	{
	public:
		BestFirstBeamSearchProcedure();
		BestFirstBeamSearchProcedure(int beamSize);
		~BestFirstBeamSearchProcedure();

		virtual vector< ISearchNode* > selectSubsetOpenSet(SearchNodeHeuristicPQ& openSet);
		virtual SearchNodeHeuristicPQ expandElements(vector< ISearchNode* > subsetOpenSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet);
	};

	/**************** Greedy Procedure ****************/

	/*!
	 * @brief Greedy search procedure.
	 */
	class GreedySearchProcedure : public BreadthFirstBeamSearchProcedure
	{
	public:
		GreedySearchProcedure();
		~GreedySearchProcedure();
	};

	/*! @} */



	/**************** Search Node ****************/

	class ISearchProcedure::ISearchNode
	{
	protected:
		/*!
		 * Pointer to parent node
		 */
		ISearchNode* parent;

		/*!
		 * // Structured features of node
		 */
		ImgFeatures* X;

		/*!
		 * Structured labeling of node
		 */
		ImgLabeling YPred;

		/*!
		 * Pointer to search space
		 */
		SearchSpace* searchSpace;

	public:
		virtual ~ISearchNode() {}
		
		/*!
		 * Generate successor nodes.
		 */
		vector< ISearchNode* > generateSuccessorNodes();

		/*!
		 * Get the heuristic features of the node. 
		 * May not be defined depending on search type.
		 */
		virtual RankFeatures getHeuristicFeatures();

		/*!
		 * Get the cost features of the node. 
		 * May not be defined depending on search type.
		 */
		virtual RankFeatures getCostFeatures();

		/*!
		 * Get the heuristic value. 
		 * (From heuristic weights or loss function depending on search type.)
		 */
		virtual double getHeuristic()=0;

		/*!
		 * Get the cost value. 
		 * (From cost weights or loss function depending on search type.)
		 */
		virtual double getCost()=0;

		/*!
		 * Get the labeling of the node.
		 */
		ImgLabeling getY();

	protected:
		/*!
		 * Return type of search node.
		 */
		virtual SearchType getType()=0;
	};

	/**************** LL Search Node ****************/

	class ISearchProcedure::LLSearchNode : public ISearchNode
	{
	protected:
		/*!
		 * Pointer to groundtruth labeling
		 */
		ImgLabeling* YTruth;

		/*!
		 * Loss value
		 */
		double loss;

	public:
		LLSearchNode();

		/*!
		 * Constructor for initial state
		 */
		LLSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace);
		
		/*!
		 * Constructor for non-initial state
		 */
		LLSearchNode(ISearchNode* parent, ImgLabeling YPred);

		virtual double getHeuristic();
		virtual double getCost();
	
	protected:
		virtual SearchType getType();
	};

	/**************** HL Search Node ****************/

	class ISearchProcedure::HLSearchNode : public ISearchNode
	{
	protected:
		/*!
		 * Pointer to groundtruth labeling
		 */
		ImgLabeling* YTruth;

		/*!
		 * Heuristic features of node
		 */
		RankFeatures heuristicFeatures;

		/*!
		 * Heuristic model
		 */
		IRankModel* heuristicModel;

		/*!
		 * Heuristic value
		 */
		double heuristic;

		/*!
		 * Loss value
		 */
		double loss;

	public:
		HLSearchNode();

		/*!
		 * Constructor for initial state
		 */
		HLSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace, IRankModel* heuristicModel);
		
		/*!
		 * Constructor for non-initial state
		 */
		HLSearchNode(ISearchNode* parent, ImgLabeling YPred);

		virtual RankFeatures getHeuristicFeatures();
		virtual double getHeuristic();
		virtual double getCost();

	protected:
		virtual SearchType getType();
	};

	/**************** LC Search Node ****************/

	class ISearchProcedure::LCSearchNode : public ISearchNode
	{
	protected:
		/*!
		 * Pointer to groundtruth labeling
		 */
		ImgLabeling* YTruth;

		/*!
		 * Cost features features of node
		 */
		RankFeatures costFeatures;

		/*!
		 * Cost model
		 */
		IRankModel* costModel;

		/*!
		 * Cost value
		 */
		double cost;

		/*!
		 * Loss value
		 */
		double loss;

	public:
		LCSearchNode();

		/*!
		 * Constructor for initial state
		 */
		LCSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace, IRankModel* costModel);
		
		/*!
		 * Constructor for non-initial state
		 */
		LCSearchNode(ISearchNode* parent, ImgLabeling YPred);

		virtual RankFeatures getCostFeatures();
		virtual double getHeuristic();
		virtual double getCost();

	protected:
		virtual SearchType getType();
	};

	/**************** HC Search Node ****************/

	class ISearchProcedure::HCSearchNode : public ISearchNode
	{
	protected:
		/*!
		 * Heuristic features of node
		 */
		RankFeatures heuristicFeatures;

		/*!
		 * Cost features features of node
		 */
		RankFeatures costFeatures;
		
		/*!
		 * Heuristic model
		 */
		IRankModel* heuristicModel;

		/*!
		 * Cost model
		 */
		IRankModel* costModel;

		/*!
		 * Heuristic value
		 */
		double heuristic;

		/*!
		 * Cost value
		 */
		double cost;

	public:
		HCSearchNode();

		/*!
		 * Constructor for initial state
		 */
		HCSearchNode(ImgFeatures* X, SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel);
		
		/*!
		 * Constructor for non-initial state
		 */
		HCSearchNode(ISearchNode* parent, ImgLabeling YPred);

		virtual RankFeatures getHeuristicFeatures();
		virtual RankFeatures getCostFeatures();
		virtual double getHeuristic();
		virtual double getCost();

	protected:
		virtual SearchType getType();
	};

	/**************** Learn H Search Node ****************/

	class ISearchProcedure::LearnHSearchNode : public LLSearchNode
	{
	protected:
		/*!
		 * Heuristic features of node
		 */
		RankFeatures heuristicFeatures;

	public:
		LearnHSearchNode();

		/*!
		 * Constructor for initial state
		 */
		LearnHSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace);
		
		/*!
		 * Constructor for non-initial state
		 */
		LearnHSearchNode(ISearchNode* parent, ImgLabeling YPred);

	protected:
		virtual RankFeatures getHeuristicFeatures();
		virtual SearchType getType();
	};

	/**************** Learn C Search Node ****************/

	class ISearchProcedure::LearnCSearchNode : public HLSearchNode
	{
	protected:
		/*!
		 * Cost features features of node
		 */
		RankFeatures costFeatures;

	public:
		LearnCSearchNode();

		/*!
		 * Constructor for initial state
		 */
		LearnCSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace, IRankModel* heuristicModel);
		
		/*!
		 * Constructor for non-initial state
		 */
		LearnCSearchNode(ISearchNode* parent, ImgLabeling YPred);

	protected:
		virtual RankFeatures getCostFeatures();
		virtual SearchType getType();
	};

	/**************** Learn C Given Oracle H Search Node ****************/

	class ISearchProcedure::LearnCOracleHSearchNode : public LLSearchNode
	{
	protected:
		/*!
		 * Cost features features of node
		 */
		RankFeatures costFeatures;

	public:
		LearnCOracleHSearchNode();

		/*!
		 * Constructor for initial state
		 */
		LearnCOracleHSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace);
		
		/*!
		 * Constructor for non-initial state
		 */
		LearnCOracleHSearchNode(ISearchNode* parent, ImgLabeling YPred);

	protected:
		virtual RankFeatures getCostFeatures();
		virtual SearchType getType();
	};

	/**************** Compare Search Node ****************/

	class ISearchProcedure::CompareByHeuristic
	{
	public:
		bool operator() (ISearchNode*& lhs, ISearchNode*& rhs) const;
	};

	class ISearchProcedure::CompareByCost
	{
	public:
		bool operator() (ISearchNode*& lhs, ISearchNode*& rhs) const;
	};

	/**************** Template definitions ****************/

	template <class T>
	bool IBasicSearchProcedure::isDuplicate(ISearchNode* state, T& pq)
	{
		int size = pq.size();
		bool isDuplicate = false;

		T temp;

		for (int i = 0; i < size; i++)
		{
			ISearchNode* current = pq.top();
			pq.pop();

			if (!isDuplicate && current->getY().graph.nodesData == state->getY().graph.nodesData)
			{
				isDuplicate = true;
			}

			temp.push(current);
		}

		// reset priority queue passed as argument
		pq = temp;

		return isDuplicate;
	}

	template <class T>
	void IBasicSearchProcedure::deleteQueueElements(T& queue)
	{
		while (!queue.empty())
		{
			ISearchNode* state = queue.top();
			queue.pop();
			delete state;
		}
	}
}

#endif