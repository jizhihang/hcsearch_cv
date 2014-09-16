#include "BerkeleySegmentation.h"
#include <algorithm>

namespace HCSearch
{
	BerkeleySegmentationUCM::BerkeleySegmentationUCM()
	{
		// NOT IMPLEMENTED
	}

	BerkeleySegmentationUCM::BerkeleySegmentationUCM(ImgLabeling graph, map< Edge_t, double > edgeWeights)
	{
		this->graph = graph;
		this->edgeWeights = edgeWeights;
		this->weightsList = BerkeleySegmentationUtilities::getAllUniqueUCMValues(edgeWeights);
	}

	BerkeleySegmentationUCM::~BerkeleySegmentationUCM()
	{
	}

	vector<double> BerkeleySegmentationUCM::getAllWeights()
	{
		return this->weightsList;
	}

	ImgLabeling BerkeleySegmentationUCM::getThresholdedGraph(double threshold)
	{
		// first pass: determine edges to remove
		vector<Edge_t> edgesToKeep;
		for (map< Edge_t, double >::iterator it = this->edgeWeights.begin(); it != this->edgeWeights.end(); ++it)
		{
			Edge_t edge = it->first;
			double weight = it->second;

			if (weight > threshold)
			{
				edgesToKeep.push_back(edge);
			}
		}

		// second pass: remove edges
		AdjList_t adjList;
		for (vector<Edge_t>::iterator it = edgesToKeep.begin(); it != edgesToKeep.end(); ++it)
		{
			Edge_t edge = *it;
			Node_t node1 = edge.first;
			Node_t node2 = edge.second;
			
			// keep these uncut edges
			if (adjList.count(node1) == 0)
			{
				adjList[node1] = set<int>();
			}
			if (adjList.count(node2) == 0)
			{
				adjList[node2] = set<int>();
			}
			adjList[node1].insert(node2);
			adjList[node2].insert(node1);
		}

		ImgLabeling graphCut;
		graphCut.confidences = this->graph.confidences;
		graphCut.confidencesAvailable = this->graph.confidencesAvailable;
		graphCut.graph = this->graph.graph;
		graphCut.graph.adjList = adjList;

		return graphCut;
	}

	ImgLabeling BerkeleySegmentationUCM::getGraph()
	{
		return this->graph;
	}

	map< Edge_t, double > BerkeleySegmentationUCM::getEdgeWeights()
	{
		return this->edgeWeights;
	}

	vector<double> BerkeleySegmentationUtilities::getAllUniqueUCMValues(map< Edge_t, double > edgeWeights)
	{
		set<double> w;
		for (map< Edge_t, double >::iterator it = edgeWeights.begin(); it != edgeWeights.end(); ++it)
			w.insert(it->second);

		vector<double> weightsList = vector<double>(w.begin(), w.end());
		sort(weightsList.begin(), weightsList.end());

		return weightsList;
	}
}