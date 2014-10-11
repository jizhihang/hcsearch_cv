#include "Demo.hpp"

void demo(int timeBound)
{
	// This demo appears in the Quick Start (API) guide.

	// datasets
	vector<string> trainFiles;
	vector<string> validationFiles;
	vector<string> testFiles;

	// load dataset
	HCSearch::Dataset::loadDataset(trainFiles, validationFiles, testFiles);

	// load search space functions and search space
	HCSearch::IFeatureFunction* heuristicFeatFunc = new HCSearch::StandardFeatures();
	HCSearch::IFeatureFunction* costFeatFunc = new HCSearch::StandardFeatures();
	HCSearch::IInitialPredictionFunction* logRegInitPredFunc = new HCSearch::LogRegInit();
	HCSearch::ISuccessorFunction* stochasticSuccessor = new HCSearch::StochasticSuccessor();
	HCSearch::ILossFunction* lossFunc = new HCSearch::HammingLoss();
	HCSearch::IPruneFunction* pruneFunc = new HCSearch::NoPrune();
	HCSearch::SearchSpace* searchSpace = new  HCSearch::SearchSpace(heuristicFeatFunc, costFeatFunc, logRegInitPredFunc, stochasticSuccessor, pruneFunc, lossFunc);

	// load search procedure
	HCSearch::ISearchProcedure* searchProcedure = new HCSearch::GreedySearchProcedure();

	// set rank learner algorithm
	HCSearch::RankerType ranker = HCSearch::VW_RANK;

	// train H
	HCSearch::IRankModel* heuristicModel = HCSearch::Learning::learnH(trainFiles, validationFiles, 
	timeBound, searchSpace, searchProcedure, ranker, 1);

	// train C
	HCSearch::IRankModel* costModel = HCSearch::Learning::learnC(trainFiles, validationFiles, 
	heuristicModel, timeBound, searchSpace, searchProcedure, ranker, 1);

	// run HC search inference on the first test example for demo
	HCSearch::ImgFeatures* XTestObj = NULL;
	HCSearch::ImgLabeling* YTestObj = NULL;
	HCSearch::Dataset::loadImage(testFiles[0], XTestObj, YTestObj);

	HCSearch::ISearchProcedure::SearchMetadata searchMetadata; // no meta data needed for this demo
	HCSearch::Inference::runHCSearch(XTestObj, timeBound, searchSpace, searchProcedure, heuristicModel, costModel, searchMetadata);

	HCSearch::Dataset::unloadImage(XTestObj, YTestObj);

	// save models for later use
	HCSearch::Model::saveModel(heuristicModel, "path/to/heuristic/model.txt", ranker);
	HCSearch::Model::saveModel(costModel, "path/to/cost/model.txt", ranker);

	// clean up
	delete searchSpace;
	delete searchProcedure;
	delete heuristicModel;
	delete costModel;
}
