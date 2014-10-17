#include <iostream>
#include <ctime>
#include "HCSearch.hpp"
#include "MyFileSystem.hpp"

using namespace std;

namespace HCSearch
{
	/**************** Initialize/Finalize ****************/

	void Setup::initialize(int argc, char* argv[])
	{
#ifdef USE_MPI
		// initialize MPI
		int rank, size;
		int rc = MPI_Init(&argc, &argv);
		if (rc != MPI_SUCCESS)
		{
			LOG(ERROR) << "error starting MPI program. Terminating.";
			exit(1);
		}

		// get size and rank
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		// initialize settings/logger
		initializeHelper();

		// set rank and number of processes
		Global::settings->RANK = rank;
		Global::settings->NUM_PROCESSES = size;
		Global::settings->MPI_STATUS = new MPI_Status();

		LOG() << "Process [" << Global::settings->RANK << "/" 
			<< Global::settings->NUM_PROCESSES 
			<< "]: MPI initialized!" << endl << endl;
#else
		initializeHelper();
#endif
	}

	void Setup::configure(string datasetPath, string outputPath, string basePath)
	{
		// refresh Settings
		Global::settings->refresh(MyFileSystem::FileSystem::normalizeDirString(datasetPath), 
			MyFileSystem::FileSystem::normalizeDirString(outputPath),
			MyFileSystem::FileSystem::normalizeDirString(basePath));

		// create output folders
		MyFileSystem::FileSystem::createFolder(Global::settings->paths->OUTPUT_DIR);
		MyFileSystem::FileSystem::createFolder(Global::settings->paths->OUTPUT_TEMP_DIR);
		MyFileSystem::FileSystem::createFolder(Global::settings->paths->OUTPUT_RESULTS_DIR);
		MyFileSystem::FileSystem::createFolder(Global::settings->paths->OUTPUT_LOGS_DIR);
		MyFileSystem::FileSystem::createFolder(Global::settings->paths->OUTPUT_MODELS_DIR);

		// set up logging
		MyLogger::Logger::initialize(Global::settings->RANK, Global::settings->NUM_PROCESSES, Global::settings->paths->OUTPUT_LOG_FILE);

		// set classes
		setClasses();
	}

	void Setup::finalize()
	{
#ifdef USE_MPI
		EasyMPI::EasyMPI::synchronize("DONESTART", "DONEEND");

		finalizeHelper();

		LOG() << "Process [" << Global::settings->RANK << "/" 
			<< Global::settings->NUM_PROCESSES 
			<< "] is DONE and exiting..." << endl;
		MPI_Finalize();
#else
		finalizeHelper();
#endif
	}

	void Setup::initializeHelper()
	{
		LOG() << "Initializing HCSearch... ";

		// initialize settings
		Global::settings = new Settings();

		LOG() << "done!" << endl << endl;
	}

	void Setup::finalizeHelper()
	{
		if (Global::settings != NULL)
			delete Global::settings;

		MyLogger::Logger::finalize();
	}

	void Setup::setClasses()
	{
		set<int> allClassesSet, backgroundClassesSet, foregroundClassesSet;
		int backgroundLabel;
		bool foundBackgroundLabel = false;
		string line;
		string filename = Global::settings->paths->INPUT_METADATA_FILE;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			while (fh.good())
			{
				getline(fh, line);
				stringstream ss(line);
				string exampletype;
				string list;
				getline(ss, exampletype, '=');
				getline(ss, list, '=');
				if (exampletype.compare("classes") == 0)
				{
					allClassesSet = parseList(list);
				}
				else if (exampletype.compare("backgroundclasses") == 0)
				{
					backgroundClassesSet = parseList(list);
				}
				else if (exampletype.compare("backgroundlabel") == 0)
				{
					stringstream ss(list);
					string num;
					getline(ss, num, ',');
					backgroundLabel = atoi(num.c_str());
					foundBackgroundLabel = true;
				}
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open meta file for reading!";
		}

		// process class index and labels
		set_difference(allClassesSet.begin(), allClassesSet.end(), 
			backgroundClassesSet.begin(), backgroundClassesSet.end(), 
			inserter(foregroundClassesSet, foregroundClassesSet.end()));

		LOG() << "=== Class Statistics ===" << endl;
		LOG() << "Class Labels: ";
		int classIndex = 0;
		for (set<int>::iterator it = foregroundClassesSet.begin(); it != foregroundClassesSet.end(); ++it)
		{
			int label = *it;
			Global::settings->CLASSES.addClass(classIndex, label, false);
			LOG() << label << ", ";
			classIndex++;
		}
		LOG() << endl;
		LOG() << "Background Class Labels: ";
		for (set<int>::iterator it = backgroundClassesSet.begin(); it != backgroundClassesSet.end(); ++it)
		{
			int label = *it;
			Global::settings->CLASSES.addClass(classIndex, label, true);
			LOG() << label << ", ";
			classIndex++;
		}
		LOG() << endl;
		if (foundBackgroundLabel)
		{
			Global::settings->CLASSES.setBackgroundLabel(backgroundLabel);
			LOG() << "Main Background Label: " << backgroundLabel << endl;
		}
		else if (backgroundClassesSet.size() == 1)
		{
			for (set<int>::iterator it = backgroundClassesSet.begin(); it != backgroundClassesSet.end(); ++it)
			{
				int label = *it;
				Global::settings->CLASSES.setBackgroundLabel(label);
				LOG() << "Main Background Class: " << label << endl;
			}
		}
		LOG() << endl;
	}

	set<int> Setup::parseList(string str)
	{
		set<int> list = set<int>();

		if (!str.empty())
		{
			stringstream ss(str);
			string num;
			while (getline(ss, num, ','))
			{
				list.insert(atoi(num.c_str()));
			}
		}

		return list;
	}

	/**************** Dataset ****************/

	void Dataset::loadDataset(vector<string>& trainFiles, vector<string>& validationFiles, vector<string>& testFiles)
	{
		LOG() << "=== Loading Dataset ===" << endl;

		// read in training data
		string trainSplitFile = Global::settings->paths->INPUT_SPLITS_TRAIN_FILE;
		LOG() << endl << "Reading from " << trainSplitFile << "..." << endl;
		trainFiles = readSplitsFile(trainSplitFile);

		// read in validation data
		string validSplitFile = Global::settings->paths->INPUT_SPLITS_VALIDATION_FILE;
		LOG() << endl << "Reading from " << validSplitFile << "..." << endl;
		validationFiles = readSplitsFile(validSplitFile);

		// read in test data
		string testSplitFile = Global::settings->paths->INPUT_SPLITS_TEST_FILE;
		LOG() << endl << "Reading from " << testSplitFile << "..." << endl;
		testFiles = readSplitsFile(testSplitFile);

		LOG() << endl;
	}

	void Dataset::loadImage(string fileName, ImgFeatures*& X, ImgLabeling*& Y)
	{
		LOG() << "\tLoading " << fileName << "..." << endl;

		// read meta file
		string metaFile = Global::settings->paths->INPUT_META_DIR + fileName + ".txt";
		int numNodes, numFeatures, height, width;
		readMetaFile(metaFile, numNodes, numFeatures, height, width);

		// read nodes file
		string nodesFile = Global::settings->paths->INPUT_NODES_DIR + fileName + ".txt";
		VectorXi labels = VectorXi::Zero(numNodes);
		MatrixXd features = MatrixXd::Zero(numNodes, numFeatures);
		readNodesFile(nodesFile, labels, features);

		// read node locations
		string nodeLocationsFile = Global::settings->paths->INPUT_NODE_LOCATIONS_DIR + fileName + ".txt";
		MatrixXd nodeLocations = MatrixXd::Zero(numNodes, 2);
		VectorXd nodeWeights = VectorXd::Zero(numNodes);
		readNodeLocationsFile(nodeLocationsFile, nodeLocations, nodeWeights);

		// read edges file
		string edgesFile = Global::settings->paths->INPUT_EDGES_DIR + fileName + ".txt";
		AdjList_t edges;
		map< MyPrimitives::Pair<int, int>, double > edgeWeights;
		readEdgesFile(edgesFile, edges, edgeWeights);

		// read segments file
		string segmentsFile = Global::settings->paths->INPUT_SEGMENTS_DIR + fileName + ".txt";
		MatrixXi segments = MatrixXi::Zero(height, width);
		readSegmentsFile(segmentsFile, segments);

		// construct ImgFeatures
		FeatureGraph featureGraph;
		featureGraph.adjList = edges;
		featureGraph.nodesData = features;
		X = new ImgFeatures();
		X->graph = featureGraph;
		X->filename = fileName;
		X->segmentsAvailable = true;
		X->segments = segments;
		X->nodeLocationsAvailable = true;
		X->nodeLocations = nodeLocations;
		X->edgeWeightsAvailable = Global::settings->USE_EDGE_WEIGHTS;
		X->edgeWeights = edgeWeights;

		// construct ImgLabeling
		LabelGraph labelGraph;
		labelGraph.adjList = edges;
		labelGraph.nodesData = labels;
		Y = new ImgLabeling();
		Y->graph = labelGraph;
		Y->nodeWeightsAvailable = true;
		Y->nodeWeights = nodeWeights;
	}

	void Dataset::unloadImage(ImgFeatures*& X, ImgLabeling*& Y)
	{
		delete X;
		delete Y;
		X = NULL;
		Y = NULL;
	}

	void Dataset::computeTaskRange(int rank, int numTasks, int numProcesses, int& start, int& end)
	{
		if (rank >= numTasks)
		{
			start = 0;
			end = 0;
		}
		else
		{
			if (rank < numTasks%numProcesses)
			{
				start = (int)( rank*ceil(1.0*numTasks/numProcesses) );
				end = (int)( (rank+1)*ceil(1.0*numTasks/numProcesses) );
			}
			else
			{
				start = (int)( (numTasks%numProcesses)*ceil(1.0*numTasks/numProcesses) + (rank - numTasks%numProcesses)*floor(1.0*numTasks/numProcesses) );
				end = (int)( (numTasks%numProcesses)*ceil(1.0*numTasks/numProcesses) + (rank+1 - numTasks%numProcesses)*floor(1.0*numTasks/numProcesses) );
			}
		}
	}

	vector<string> Dataset::readSplitsFile(string filename)
	{
		vector<string> filenames;

		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					filenames.push_back(line);
				}
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open splits file!";
			abort();
		}

		return filenames;
	}

	void Dataset::readMetaFile(string filename, int& numNodes, int& numFeatures, int& height, int& width)
	{
		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			while (fh.good())
			{
				getline(fh, line);
				stringstream ss(line);
				string tag;
				string num;
				getline(ss, tag, '=');
				getline(ss, num, '=');
				if (tag.compare("nodes") == 0)
				{
					numNodes = atoi(num.c_str());
				}
				else if (tag.compare("features") == 0)
				{
					numFeatures = atoi(num.c_str());
				}
				else if (tag.compare("height") == 0)
				{
					height = atoi(num.c_str());
				}
				else if (tag.compare("width") == 0)
				{
					width = atoi(num.c_str());
				}
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open meta file!";
			abort();
		}
	}

	void Dataset::readNodesFile(string filename, VectorXi& labels, MatrixXd& features)
	{
		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			int lineIndex = 0;
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					if (lineIndex >= labels.size())
					{
						LOG(WARNING) << "line index exceeds number of nodes; ignoring the rest...";
						break;
					}

					// parse line
					istringstream iss(line);
					string token;

					// get label
					getline(iss, token, ' ');
					labels(lineIndex) = atoi(token.c_str());

					// get features
					while (getline(iss, token, ' '))
					{
						if (!token.empty())
						{
							istringstream iss2(token);
							string sIndex;
							getline(iss2, sIndex, ':');
							string sValue;
							getline(iss2, sValue, ':');

							int featureIndex = atoi(sIndex.c_str()) - 1;
							double value = atof(sValue.c_str());

							if (featureIndex < 0 || featureIndex >= features.cols())
							{
								LOG(WARNING) << "feature index exceeds number of features; ignoring...";
							}
							else
							{
								features(lineIndex, featureIndex) = value;
							}
						}
					}
				}
				lineIndex++;
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file to nodes data!";
			abort();
		}
	}

	void Dataset::readNodeLocationsFile(string filename, MatrixXd& nodeLocations, VectorXd& nodeWeights)
	{
		int totalSize = 0;
		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			int lineIndex = 0;
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					if (lineIndex >= nodeLocations.rows())
					{
						LOG(WARNING) << "line index exceeds number of nodes; ignoring the rest...";
						break;
					}

					// parse line
					istringstream iss(line);

					// get x position
					string token1;
					getline(iss, token1, ' ');
					nodeLocations(lineIndex, 0) = atof(token1.c_str());

					// get y position
					string token2;
					getline(iss, token2, ' ');
					nodeLocations(lineIndex, 1) = atof(token2.c_str());

					// get segment size
					string token3;
					getline(iss, token3, ' ');
					int size = atoi(token3.c_str());
					nodeWeights(lineIndex) = size;
					totalSize += size;
				}
				lineIndex++;
			}
			fh.close();

			// normalize segment sizes
			nodeWeights /= (1.0*totalSize);
		}
		else
		{
			LOG(ERROR) << "cannot open file to node locations data!";
			abort();
		}
	}

	void Dataset::readEdgesFile(string filename, AdjList_t& edges, map< MyPrimitives::Pair<int, int>, double >& edgeWeights)
	{
		// if 1, then node indices in edge file are 1-based
		// if 0, then node indices in edge file are 0-based
		const int ONE_OFFSET = 1;

		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			// current line = current node
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					// parse line
					istringstream iss(line);

					// get node1
					string token1;
					getline(iss, token1, ' ');
					int node1 = atoi(token1.c_str()) - ONE_OFFSET;

					// get node2
					string token2;
					getline(iss, token2, ' ');
					int node2 = atoi(token2.c_str()) - ONE_OFFSET;

					// get 1 (or weight)
					string token3;
					getline(iss, token3, ' ');
					double edgeWeight = atof(token3.c_str());

					// add to map
					if (edges.count(node1) == 0)
					{
						edges[node1] = set<int>();
					}
					edges[node1].insert(node2);

					// add to edge weights
					MyPrimitives::Pair<int, int> edge = MyPrimitives::Pair<int, int>(node1, node2);
					edgeWeights[edge] = edgeWeight;
				}
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file to edges data!";
			abort();
		}
	}

	void Dataset::readSegmentsFile(string filename, MatrixXi& segments)
	{
		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			// current line
			int lineIndex = 0;
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					// parse line
					istringstream iss(line);
					string token;

					// current column
					int columnIndex = 0;
					while (getline(iss, token, ' '))
					{
						if (!token.empty())
						{
							int value = atoi(token.c_str());
							segments(lineIndex, columnIndex) = value;
						}
						columnIndex++;
					}
				}
				lineIndex++;
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file to segments data!";
			abort();
		}
	}

	/**************** Model ****************/

	IRankModel* Model::loadModel(string fileName, RankerType rankerType)
	{
		if (rankerType == SVM_RANK)
		{
			SVMRankModel* model = new SVMRankModel();
			model->load(fileName);
			return model;
		}
		else if (rankerType == VW_RANK)
		{
			VWRankModel* model = new VWRankModel();
			model->load(fileName);
			return model;
		}
		else if (rankerType == VW_RANK)
		{
			VWRankModel* model = new VWRankModel();
			model->load(fileName);
			return model;
		}
		else
		{
			LOG(ERROR) << "ranker type is invalid for loading model";
			return NULL;
		}
	}

	map<string, int> Model::loadPairwiseConstraints(string fileName)
	{
		map<string, int> pairwiseConstraints;

		//TODO
		int lineIndex = 0;
		string line;
		ifstream fh(fileName.c_str());
		if (fh.is_open())
		{
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					// parse line
					stringstream ss(line);
					string token;
					int columnIndex = 0;
					int class1, class2, counts;
					string configuration;
					while (getline(ss, token, ' '))
					{
						if (columnIndex == 0)
						{
							class1 = atoi(token.c_str());
						}
						else if (columnIndex == 1)
						{
							class2 = atoi(token.c_str());
						}
						else if (columnIndex == 2)
						{
							configuration = token.c_str();
						}
						else if (columnIndex == 3)
						{
							counts = atoi(token.c_str());
						}
						columnIndex++;
					}
					if (columnIndex < 4)
					{
						LOG(ERROR) << "parsing illegal format for pairwise discovery";
						abort();
					}

					stringstream configSS;
					configSS << class1 << " " << class2 << " " << configuration;
					string configString = configSS.str();
					if (pairwiseConstraints.count(configString) == 0)
					{
						pairwiseConstraints[configString] = counts;
					}
					else
					{
						LOG(WARNING) << "configuration string already in map!";
					}
				}

				lineIndex++;
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file for reading pairwise constraints!";
			abort();
		}

		return pairwiseConstraints;
	}

	void Model::saveModel(IRankModel* model, string fileName, RankerType rankerType)
	{
		if (model == NULL)
		{
			LOG(ERROR) << "rank model is NULL, so cannot save it.";
			return;
		}

		if (rankerType == SVM_RANK)
		{
			SVMRankModel* modelCast = dynamic_cast<SVMRankModel*>(model);
			modelCast->save(fileName);
		}
		else if (rankerType == VW_RANK)
		{
			VWRankModel* modelCast = dynamic_cast<VWRankModel*>(model);
			modelCast->save(fileName);
		}
		else
		{
			LOG(ERROR) << "ranker type is invalid for saving model";
		}
	}

	void Model::savePairwiseConstraints(map<string, int>& pairwiseConstraints, string fileName)
	{
		ofstream fh(fileName.c_str());
		if (fh.is_open())
		{
			for (map<string, int>::iterator it = pairwiseConstraints.begin(); it != pairwiseConstraints.end(); ++it)
			{
				string key = it->first;
				int value = it->second;

				fh << key << " " << value << endl;
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file for saving pairwise constraints!";
			abort();
		}
	}

	/**************** Learning ****************/

	IRankModel* Learning::learnH(vector<string>& trainFiles, vector<string>& validFiles,
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter)
	{
		clock_t tic = clock();

		LOG() << "Learning the heuristic function..." << endl;
		
		// Setup model for learning
		IRankModel* learningModel = Training::initializeLearning(rankerType, LEARN_H);

		// Learn on each training example
		// set up commands
		vector<string> commands;
		vector<string> messages;
		for (int imageID = 0; imageID < static_cast<int>(trainFiles.size()); imageID++)
		{
			for (int iter = 0; iter < numIter; iter++)
			{
				stringstream ssMessage;
				ssMessage << imageID << ":" << iter;

				commands.push_back("LEARNH");
				messages.push_back(ssMessage.str());
			}
		}

		// schedule and perform tasks
		if (HCSearch::Global::settings->RANK == 0 && HCSearch::Global::settings->NUM_PROCESSES > 1)
		{
			EasyMPI::EasyMPI::masterScheduleTasks(commands, messages);
		}
		else
		{
			runSlave(commands, messages, trainFiles, validFiles, timeBound, 
				searchSpace, searchProcedure, learningModel, NULL);
		}
		
		// Merge and learn step
		Training::finishLearning(learningModel, LEARN_H);

		clock_t toc = clock();
		LOG() << "total learnH time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;

		return learningModel;
	}

	IRankModel* Learning::learnC(vector<string>& trainFiles, vector<string>& validFiles,
		IRankModel* heuristicModel, int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter)
	{
		clock_t tic = clock();

		LOG() << "Learning the cost function with learned heuristic..." << endl;
		
		// Setup model for learning
		IRankModel* learningModel = Training::initializeLearning(rankerType, LEARN_C);

		// Learn on each training example
		// set up commands
		vector<string> commands;
		vector<string> messages;
		for (int imageID = 0; imageID < static_cast<int>(trainFiles.size()); imageID++)
		{
			for (int iter = 0; iter < numIter; iter++)
			{
				stringstream ssMessage;
				ssMessage << imageID << ":" << iter;

				commands.push_back("LEARNC");
				messages.push_back(ssMessage.str());
			}
		}

		// schedule and perform tasks
		if (HCSearch::Global::settings->RANK == 0 && HCSearch::Global::settings->NUM_PROCESSES > 1)
		{
			EasyMPI::EasyMPI::masterScheduleTasks(commands, messages);
		}
		else
		{
			runSlave(commands, messages, trainFiles, validFiles, timeBound, 
				searchSpace, searchProcedure, learningModel, heuristicModel);
		}
		
		// Merge and learn step
		Training::finishLearning(learningModel, LEARN_C);

		clock_t toc = clock();
		LOG() << "total learnC time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;

		return learningModel;
	}

	IRankModel* Learning::learnCWithOracleH(vector<string>& trainFiles, vector<string>& validFiles,
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter)
	{
		clock_t tic = clock();

		LOG() << "Learning the cost function with oracle heuristic..." << endl;

		// Setup model for learning
		IRankModel* learningModel = Training::initializeLearning(rankerType, LEARN_C_ORACLE_H);

		// Learn on each training example
		// set up commands
		vector<string> commands;
		vector<string> messages;
		for (int imageID = 0; imageID < static_cast<int>(trainFiles.size()); imageID++)
		{
			for (int iter = 0; iter < numIter; iter++)
			{
				stringstream ssMessage;
				ssMessage << imageID << ":" << iter;

				commands.push_back("LEARNCOH");
				messages.push_back(ssMessage.str());
			}
		}

		// schedule and perform tasks
		if (HCSearch::Global::settings->RANK == 0 && HCSearch::Global::settings->NUM_PROCESSES > 1)
		{
			EasyMPI::EasyMPI::masterScheduleTasks(commands, messages);
		}
		else
		{
			runSlave(commands, messages, trainFiles, validFiles, timeBound, 
				searchSpace, searchProcedure, learningModel, NULL);
		}
		
		// Merge and learn step
		Training::finishLearning(learningModel, LEARN_C_ORACLE_H);

		clock_t toc = clock();
		LOG() << "total learnCWithOracleH time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;

		return learningModel;
	}

	IRankModel* Learning::learnP(vector<string>& trainFiles, vector<string>& validFiles,
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter)
	{
		clock_t tic = clock();

		LOG() << "Learning the prune function..." << endl;

		// Setup model for learning
		IRankModel* learningModel = Training::initializeLearning(rankerType, LEARN_PRUNE);

		// Learn on each training example
		// set up commands
		vector<string> commands;
		vector<string> messages;
		for (int imageID = 0; imageID < static_cast<int>(trainFiles.size()); imageID++)
		{
			for (int iter = 0; iter < numIter; iter++)
			{
				stringstream ssMessage;
				ssMessage << imageID << ":" << iter;

				commands.push_back("LEARNP");
				messages.push_back(ssMessage.str());
			}
		}

		// schedule and perform tasks
		if (HCSearch::Global::settings->RANK == 0 && HCSearch::Global::settings->NUM_PROCESSES > 1)
		{
			EasyMPI::EasyMPI::masterScheduleTasks(commands, messages);
		}
		else
		{
			runSlave(commands, messages, trainFiles, validFiles, timeBound, 
				searchSpace, searchProcedure, learningModel, NULL);
		}
		
		// Merge and learn step
		Training::finishLearning(learningModel, LEARN_PRUNE);

		clock_t toc = clock();
		LOG() << "total learnP time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;

		return learningModel;
	}

	map<string, int> Learning::discoverPairwiseClassConstraints(vector<string>& trainFiles)
	{
		map<string, int> pairwiseConstraints;

		clock_t tic = clock();

		LOG() << "Discovering pairwise class constraints..." << endl;

		// Learn on each training example
		int start, end;
		HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, trainFiles.size(), 
			HCSearch::Global::settings->NUM_PROCESSES, start, end);
		for (int i = start; i < end; i++)
		{
			LOG() << "Pairwise class constraint: processing on " << trainFiles[i] << " (example " << i << ")..." << endl;

			// do stuff
			ImgFeatures* X = NULL;
			ImgLabeling* Y = NULL;
			Dataset::loadImage(trainFiles[i], X, Y);
			
			const int numNodes = X->getNumNodes();
			for (int node1 = 0; node1 < numNodes; node1++)
			{
				for (int node2 = 0; node2 < numNodes; node2++)
				{
					int node1Class = Y->getLabel(node1);
					int node2Class = Y->getLabel(node2);

					if (node1 == node2 || node1Class == node2Class)
						continue;

					double node1XCoord = X->getNodeLocationX(node1);
					double node1YCoord = X->getNodeLocationY(node1);
					double node2XCoord = X->getNodeLocationX(node2);
					double node2YCoord = X->getNodeLocationY(node2);

					// check left/right
					if (node1XCoord != node2XCoord)
					{
						stringstream configSSLR;
						configSSLR << node1Class << " " << node2Class << " ";
						if (node1XCoord < node2XCoord)
						{
							// node 1 to the left of node 2
							configSSLR << "L";
						}
						else if (node1XCoord > node2XCoord)
						{
							// node 1 to the right of node 2
							configSSLR << "R";
						}
						string configStringLR = configSSLR.str();
						if (pairwiseConstraints.count(configStringLR) == 0)
						{
							pairwiseConstraints[configStringLR] = 0;
						}
						pairwiseConstraints[configStringLR]++;
					}

					// check top/bottom
					if (node1YCoord != node2YCoord)
					{
						stringstream configSSUD;
						configSSUD << node1Class << " " << node2Class << " ";
						if (node1YCoord < node2YCoord)
						{
							// node 1 above node 2
							configSSUD << "U";
						}
						else if (node1YCoord > node2YCoord)
						{
							// node 1 below node 2
							configSSUD << "D";
						}
						string configStringUD = configSSUD.str();
						if (pairwiseConstraints.count(configStringUD) == 0)
						{
							pairwiseConstraints[configStringUD] = 0;
						}
						pairwiseConstraints[configStringUD]++;
					}
				}
			}

			Dataset::unloadImage(X, Y);
		}
		
		clock_t toc = clock();
		LOG() << "total discoverPairwiseClassConstraints time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;

		return pairwiseConstraints;
	}

	void Learning::runSlave(vector<string> commands, vector<string> messages,
			vector<string>& trainFiles, vector<string>& validFiles,
			int timeBound, SearchSpace*& searchSpace, ISearchProcedure*& searchProcedure,
			IRankModel*& learningModel, IRankModel* heuristicModel)
	{
		vector<string> commandSet = commands;
		vector<string> messageSet = messages;

		// loop to wait for tasks
		while (true)
		{
			// wait for a task
			std::string command;
			std::string message;

			// wait for task if more than one process
			// otherwise if only one process, then perform task on master process
			if (HCSearch::Global::settings->NUM_PROCESSES > 1)
			{
				EasyMPI::EasyMPI::slaveWaitForTasks(command, message);
			}
			else
			{
				if (commandSet.empty() || messageSet.empty())
					break;

				command = commandSet.back();
				message = messageSet.back();
				commandSet.pop_back();
				messageSet.pop_back();
			}

			LOG(INFO) << "Got command '" << command << "' and message '" << message << "'";

			// define branches here to perform task depending on command
			if (command.compare("LEARNH") == 0)
			{

				// Declare
				int i; // image ID
				int iter; // iteration ID
				getImageIDAndIter(message, i, iter);

				LOG() << "Heuristic learning: (iter " << iter << ") beginning search on " << trainFiles[i] << " (example " << i << ")..." << endl;

				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = false;
				meta.setType = HCSearch::TRAIN;
				meta.exampleName = trainFiles[i];
				meta.iter = iter;

				ImgFeatures* XTrainObj = NULL;
				ImgLabeling* YTrainObj = NULL;
				Dataset::loadImage(trainFiles[i], XTrainObj, YTrainObj);

				// run search
				searchProcedure->performSearch(LEARN_H, *XTrainObj, YTrainObj, timeBound, searchSpace, learningModel, NULL, NULL, meta);

				Dataset::unloadImage(XTrainObj, YTrainObj);

				// declare finished
				EasyMPI::EasyMPI::slaveFinishedTask();

			}
			else if (command.compare("LEARNC") == 0)
			{

				// Declare
				int i; // image ID
				int iter; // iteration ID
				getImageIDAndIter(message, i, iter);

				LOG() << "Cost learning: (iter " << iter << ") beginning search on " << trainFiles[i] << " (example " << i << ")..." << endl;

				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = false;
				meta.setType = HCSearch::TRAIN;
				meta.exampleName = trainFiles[i];
				meta.iter = iter;

				ImgFeatures* XTrainObj = NULL;
				ImgLabeling* YTrainObj = NULL;
				Dataset::loadImage(trainFiles[i], XTrainObj, YTrainObj);

				// run search
				searchProcedure->performSearch(LEARN_C, *XTrainObj, YTrainObj, timeBound, searchSpace, heuristicModel, learningModel, NULL, meta);

				Dataset::unloadImage(XTrainObj, YTrainObj);

				// declare finished
				EasyMPI::EasyMPI::slaveFinishedTask();

			}
			else if (command.compare("LEARNCOH") == 0)
			{

				// Declare
				int i; // image ID
				int iter; // iteration ID
				getImageIDAndIter(message, i, iter);

				LOG() << "Cost with oracle H learning: (iter " << iter << ") beginning search on " << trainFiles[i] << " (example " << i << ")..." << endl;

				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = false;
				meta.setType = HCSearch::TRAIN;
				meta.exampleName = trainFiles[i];
				meta.iter = iter;

				ImgFeatures* XTrainObj = NULL;
				ImgLabeling* YTrainObj = NULL;
				Dataset::loadImage(trainFiles[i], XTrainObj, YTrainObj);

				// run search
				searchProcedure->performSearch(LEARN_C_ORACLE_H, *XTrainObj, YTrainObj, timeBound, searchSpace, NULL, learningModel, NULL, meta);

				Dataset::unloadImage(XTrainObj, YTrainObj);

				// declare finished
				EasyMPI::EasyMPI::slaveFinishedTask();

			}
			else if (command.compare("LEARNP") == 0)
			{

				// Declare
				int i; // image ID
				int iter; // iteration ID
				getImageIDAndIter(message, i, iter);

				LOG() << "Prune learning: (iter " << iter << ") beginning search on " << trainFiles[i] << " (example " << i << ")..." << endl;

				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = false;
				meta.setType = HCSearch::TRAIN;
				meta.exampleName = trainFiles[i];
				meta.iter = iter;

				ImgFeatures* XTrainObj = NULL;
				ImgLabeling* YTrainObj = NULL;
				Dataset::loadImage(trainFiles[i], XTrainObj, YTrainObj);

				// run search
				searchProcedure->performSearch(LEARN_PRUNE, *XTrainObj, YTrainObj, timeBound, searchSpace, NULL, NULL, learningModel, meta);

				Dataset::unloadImage(XTrainObj, YTrainObj);

				// declare finished
				EasyMPI::EasyMPI::slaveFinishedTask();

			}
			else if (command.compare(EasyMPI::EasyMPI::MASTER_FINISH_MESSAGE) == 0)
			{
				LOG(INFO) << "Got the master finish command on process " << HCSearch::Global::settings->RANK
					<< ". Exiting slave loop...";

				break;
			}
			else
			{
				LOG(WARNING) << "Invalid command: " << command;
			}
		}
	}

	void Learning::getImageIDAndIter(string message, int& imageID, int& iterID)
	{
		string imageIDString;
		string iterIDString;
		stringstream ss(message);
		getline(ss, imageIDString, ':');
		getline(ss, iterIDString, ':');

		imageID = atoi(imageIDString.c_str());
		iterID = atoi(iterIDString.c_str());
	}

	/**************** Inference ****************/

	ImgLabeling Inference::runLLSearch(ImgFeatures* X, ImgLabeling* YTruth, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, 
		ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->performSearch(LL, *X, YTruth, timeBound, 
			searchSpace, NULL, NULL, NULL, searchMetadata);
	}

	ImgLabeling Inference::runHLSearch(ImgFeatures* X, ImgLabeling* YTruth, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
		IRankModel* heuristicModel, ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->performSearch(HL, *X, YTruth, timeBound, 
			searchSpace, heuristicModel, NULL, NULL, searchMetadata);
	}

	ImgLabeling Inference::runLCSearch(ImgFeatures* X, ImgLabeling* YTruth, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
		IRankModel* costOracleHModel, ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->performSearch(LC, *X, YTruth, timeBound, 
			searchSpace, NULL, costOracleHModel, NULL, searchMetadata);
	}

	ImgLabeling Inference::runHCSearch(ImgFeatures* X, int timeBound, 
		SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
		IRankModel* heuristicModel, IRankModel* costModel, 
		ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->performSearch(HC, *X, NULL, timeBound, 
			searchSpace, heuristicModel, costModel, NULL, searchMetadata);
	}

	ImgLabeling Inference::runHCSearch(ImgFeatures* X, ImgLabeling* YTruth, int timeBound, 
		SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
		IRankModel* heuristicModel, IRankModel* costModel, 
		ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->performSearch(HC, *X, YTruth, timeBound, 
			searchSpace, heuristicModel, costModel, NULL, searchMetadata);
	}
}