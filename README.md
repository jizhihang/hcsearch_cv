HC-SEARCH::CV
=============

HC-Search Structured Prediction Framework for Computer Vision

## Introduction

**Thank you for your interest in this work. If you are here because of my CVPR 2015 paper, please note that this repository was actually created for a different project at the time. You can take a look at how things were overall designed, but to run your own experiments you will have to extract your own features, design your own functions and possibly implement some other functionality.**

HC-Search is a learning and inference framework for structured prediction. This is an implementation of HC-Search primarily geared toward scene labeling for computer vision, but may be adapted to other computer vision tasks.

The HC-Search framework allows you to define a search space and search procedure to perform structured prediction. HC-Search learns a heuristic function to uncover high quality candidates and a cost function to select the best candidate.

A search space consists of a heuristic feature function, cost feature function, initial state function, successor function and loss function. A search procedure can be anything that you define, including the commonly used greedy search and beam search. Our implementation allows you to define all of these components.

## Overview

Usually the following is the pipeline:

Input: Images, groundtruth annotations, and training/validation/test splits

1. (Preprocessing Modules) Extract features from images and put into format ready for the HC-Search program.
2. (HC-Search Training) Run HC-Search to learn models.
3. (HC-Search Testing) Run HC-Search to perform inference.
4. (Postprocessing Modules) Convert result files from HC-Search into format suitable for visualization, evaluation and further analysis.

Output: Evaluation, visualization and analysis of inference on test images

The next two sections explains how to get started. You can run the HCSearch application directly or use the API for your C++ programs.

## Quick Start (Application)

 This section shows how to quickly get started if you just want to use the binary executable and not the API. **This is sufficient if you only need to use the built-in search spaces and search procedures.** This section will walk through a sample scenario. Let `$ROOT$` denote the root directory containing `src`. Please first read the Installation Instructions section.

### Setup Images, Groundtruth and Train/Validation/Test Splits

1. Create a folder `$ROOT$/DataRaw/SomeDataset`. Create additional folders in it: `Annotations`, `Images` and `Splits`
2. Put the images (.jpg) in the Images/ folder and the groundtruth masks (.jpg) in the Annotations folder. Corresponding images and groundtruth files must have the same file name!
3. Create `Train.txt`, `Validation.txt` and `Test.txt` in `Splits`. In each file, list the file name in the Images folder (without .jpg extension) that belong in each split you want.

### Preprocessing

1. Create folder `$ROOT$/DataPreprocessed`.
2. Open MATLAB, make sure VLFeat is set up properly (run vl_setup), and run the following command in MATLAB. This should create files and folders in the `$ROOT$/DataPreprocessed/SomeDataset` folder.
```
preprocess('$ROOT$/DataRaw/SomeDataset/Images', '$ROOT$/DataRaw/SomeDataset/Annotations', '$ROOT$/DataRaw/SomeDataset/Splits', '$ROOT$/DataPreprocessed/SomeDataset');
```

### HC-Search Learn/Infer

Run the following command from the command line: `./HCSearch $ROOT$/DataPreprocessed/SomeDataset $ROOT$/Results/SomeExperiment 5 --learn --infer`
This learns a heuristic and cost function and then runs LL/HL/LC/HL search. 
This should create files and folders in $ROOT$/Results/

### Postprocessing

1. Create folder `$ROOT$/ResultsPreprocessed`.
2. Open in MATLAB, make sure LIBSVM is set up properly, and run the following command in MATLAB. This should create visualization files and folders in the `$ROOT$/ResultsPreprocessed/SomeExperiment` folder.
```
visualize_results('$ROOT$/DataRaw/SomeDataset', '$ROOT$/DataPreprocessed/SomeDataset', '$ROOT$/Results/SomeExperiment', '$ROOT$/ResultsPostprocessed/SomeExperiment', 0:4, 0);
```

## Quick Start (API)

This section shows how to quickly get started if you want to use the API in your C++ program. **This is necessary if you want to define your own search space and search procedure.**

The HCSearchLib is a static library. You can build it and then link it to your C++ projects.

The following is a snippet of demo code using the API. It loads the dataset, learns the heuristic and cost models, and performs inference on the first test example.

```
// initialize HCSearch
HCSearch::Setup::initialize(argc, argv);

// configure settings
HCSearch::Setup::configure("path/to/input/dir", "path/to/output/dir");

// datasets
vector< HCSearch::ImgFeatures* > XTrain;
vector< HCSearch::ImgLabeling* > YTrain;
vector< HCSearch::ImgFeatures* > XValidation;
vector< HCSearch::ImgLabeling* > YValidation;
vector< HCSearch::ImgFeatures* > XTest;
vector< HCSearch::ImgLabeling* > YTest;

// load dataset
HCSearch::Dataset::loadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);

// load search space functions and search space
HCSearch::IFeatureFunction* heuristicFeatFunc = new HCSearch::StandardFeatures();
HCSearch::IFeatureFunction* costFeatFunc = new HCSearch::StandardFeatures();
HCSearch::IInitialPredictionFunction* logRegInitPredFunc = new HCSearch::LogRegInit();
HCSearch::ISuccessorFunction* stochasticSuccessor = new HCSearch::StochasticSuccessor();
HCSearch::ILossFunction* lossFunc = new HCSearch::HammingLoss();
HCSearch::SearchSpace* searchSpace = new  HCSearch::SearchSpace(heuristicFeatFunc, costFeatFunc, logRegInitPredFunc, stochasticSuccessor, lossFunc);

// load search procedure
HCSearch::ISearchProcedure* searchProcedure = new HCSearch::GreedySearchProcedure();

// train H
HCSearch::IRankModel* heuristicModel = HCSearch::Learning::learnH(XTrain, YTrain, XValidation, YValidation, 
timeBound, searchSpace, searchProcedure, HCSearch::SVM_RANK, 1);

// train C
HCSearch::IRankModel* costModel = HCSearch::Learning::learnC(XTrain, YTrain, XValidation, YValidation, 
heuristicModel, timeBound, searchSpace, searchProcedure, HCSearch::SVM_RANK, 1);

// run HC search inference on the first test example for demo
HCSearch::ISearchProcedure::SearchMetadata searchMetadata; // no meta data needed for this demo
HCSearch::Inference::runHCSearch(XTest[0], timeBound, searchSpace, searchProcedure, heuristicModel, costModel, searchMetadata);

// save models for later use
HCSearch::Model::saveModel(heuristicModel, "path/to/heuristic/model.txt", HCSearch::SVM_RANK);
HCSearch::Model::saveModel(costModel, "path/to/cost/model.txt", HCSearch::SVM_RANK);

// clean up
HCSearch::Dataset::unloadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);

// finalize for exiting
HCSearch::Setup::finalize();
```

## Installation Instructions

Let `$ROOT$` denote the root directory containing `src` and this README.

### Windows

1. Before compiling the source or running the binary executable, dependencies must be installed in the `$ROOT$/external/` directory.
	- Eigen matrix libary
		1. Download from http://eigen.tuxfamily.org
			- Version 3.2.0 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/Eigen`
			- Unpack the directory structure such that this file path is valid: `$ROOT$/external/Eigen/Eigen/Dense`
	- SVM-Rank
		1. Download from http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
			- Make sure to download the Windows binaries version.
		2. Unpack to `$ROOT$/external/svm_rank`
			- Unpack the directory structure such that the binaries are in `$ROOT$/external/svm_rank`
	- LIBLINEAR
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
			- Version 1.94 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/liblinear`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/liblinear`
	- LIBSVM (Optional - if you use postprocessing modules OR if you use SVM for initial state in HCSearch)
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
			- Version 3.17 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/libsvm`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/libsvm`
		3. Set up in MATLAB (optional - if you use postprocessing modules):
			1. Open MATLAB, navigate in MATLAB to `$ROOT$/external/libsvm/matlab`
			2. Run the `make` command in MATLAB.
			3. Add `$ROOT$/external/libsvm/matlab` to your include paths in MATLAB.
	- VLFeat (Optional - if you use preprocessing modules)
		1. Download from http://www.vlfeat.org/
			- Version 0.9.18 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/vlfeat`
			- Unpack the directory structure such that this file path is valid: `$ROOT$/external/vlfeat/toolbox/vl_setup.m`
		3. Set up in MATLAB:
			1. Open MATLAB, add `$ROOT$/external/vlfeat/toolbox` to your include paths in MATLAB.
			2. Run the `vl_setup` command in MATLAB.
			3. Verify it is set up properly by running the `vl_version` command in MATLAB.
2. Run the provided binary executable `HCSearch.exe` (make sure it is in the top-most directory containing this README).
3. If you prefer to compile from source...
	1. Open `$ROOT$/src/HCSearch.sln` in Microsoft Visual Studio 2012 or later.
	2. Build the solution. Make sure it is on Release.
	3. Move `$ROOT$/src/Release/HCSearch.exe` to `$ROOT$/HCSearch.exe`.

### Linux

1. Before compiling the source or running the binary executable, dependencies must be installed in the `$ROOT$/external/` directory.
	- Eigen matrix libary
		1. Download from http://eigen.tuxfamily.org
			- Version 3.2.0 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/Eigen`
			- Unpack the directory structure such that this file path is valid: `$ROOT$/external/Eigen/Eigen/Dense`
	- SVM-Rank
		1. Download from http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
			- Make sure to download the source code version.
		2. Unpack to `$ROOT$/external/svm_rank`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/svm_rank`
	- LIBLINEAR
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
			- Version 1.94 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/liblinear`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/liblinear`
	- LIBSVM (Optional - if you use postprocessing modules OR if you use SVM for initial state in HCSearch)
		1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
			- Version 3.17 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/libsvm`
			- Unpack the directory structure such that the Makefile is in `$ROOT$/external/libsvm`
		3. Set up in MATLAB (optional - if you use postprocessing modules):
			1. Open MATLAB, navigate in MATLAB to `$ROOT$/external/libsvm/matlab`
			2. Run the `make` command in MATLAB.
			3. Add `$ROOT$/external/libsvm/matlab` to your include paths in MATLAB.
	- VLFeat (Optional - if you use preprocessing modules)
		1. Download from http://www.vlfeat.org/
			- Version 0.9.18 officially supported. Later versions should also work.
		2. Unpack to `$ROOT$/external/vlfeat`
			- Unpack the directory structure such that this file path is valid: `$ROOT$/external/vlfeat/toolbox/vl_setup.m`
		3. Set up in MATLAB:
			1. Open MATLAB, add `$ROOT$/external/vlfeat/toolbox` to your include paths in MATLAB.
			2. Run the `vl_setup` command in MATLAB.
			3. Verify it is set up properly by running the `vl_version` command in MATLAB.
2. Compile external dependencies by running `make all_externals` in the `$ROOT$` directory.
	- If you only want to build the required ones (SVM-Rank and LIBLINEAR), run `make externals` instead.
	- If you prefer to build manually, run `make` independently in each of the external dependency folders.
3. Compile from source by running `make` in the `$ROOT$` directory. It should create the binary file `$ROOT$/HCSearch`.

## More Details

### Preprocessing Module

The preprocessing modules are written in MATLAB. Before running them, make sure to set up VLFeat.

The main preprocessing module is `$ROOT$/preprocess/preprocess.m`:

```
function [ allData ] = preprocess( imagesPath, labelsPath, splitsPath, outputPath )
%PREPROCESS Preprocesses folders of images and groundtruth labels into a
%data format for HCSearch to work. Performs feature extraction.
%
%This implementation creates a regular grid of HOG/SIFT patches.
%Therefore only need grayscale images.
%
%One can change the features (e.g. add color) by editing this file.
%
%	imagesPath:	folder path to images folder of *.jpg images
%                   e.g. 'DataRaw/SomeDataset/Images'
%	labelsPath:	folder path to groundtruth folder of *.jpg label masks
%                   e.g. 'DataRaw/SomeDataset/Annotations'
%	splitsPath:	folder path that contains Train.txt,
%               Validation.txt, Test.txt
%                   e.g. 'DataRaw/SomeDataset/Splits'
%	outputPath:	folder path to output preprocessed data
%                   e.g. 'DataPreprocessed/SomeDataset'
%
%	allData:	data structure containing all preprocessed data
```

### HC-Search Command Line Options

```
Program usage: ./HCSearch INPUT_DIR OUTPUT_DIR TIMEBOUND [--learn (H|C|COH)]* [--infer (HC|HL|LC|LL)]* ... [--option=value]
Main options:
	--help		: produce help message
	--demo		: run the demo program (ignores --learn and --infer)
	--learn arg	: learning
				H: learn heuristic
				C: learn cost
				COH: learn cost with oracle H
				ALL: short-hand for H, C, COH
				(none): short-hand for H, C
	--infer arg	: inference
				HC: learned heuristic and cost
				HL: learned heuristic and oracle cost
				LC: oracle heuristic and learned cost
				LL: oracle heuristic and cost
				ALL: short-hand for HC, HL, LC, LL
				(none): short-hand for HC

Advanced options:
	--anytime arg			: turn on saving anytime predictions if true
	--beam-size arg			: beam size for beam search
	--cut-mode arg			: edges|state (cut edges by edges independently or by state)
	--cut-param arg			: temperature parameter for stochastic cuts
	--hfeatures arg			: standard|standard-conf|unary|unary-conf|standard-pair-counts|standard-conf-pair-counts|dense-crf
	--cfeatures arg			: standard|standard-conf|unary|unary-conf|standard-pair-counts|standard-conf-pair-counts|dense-crf
	--num-test-iters arg	: number of test iterations
	--num-train-iters arg	: number of training iterations
	--ranker arg			: svmrank|vw
	--loss arg				: hamming|pixel-hamming
	--save-features arg		: save rank features during learning if true
	--save-mask arg			: save final prediction label masks if true
	--search arg			: greedy|breadthbeam|bestbeam
	--splits-path arg		: specify alternate path to splits folder
	--splits-train-file arg	: specify alternate file name to train file
	--splits-valid-file arg	: specify alternate file name to validation file
	--splits-test-file arg	: specify alternate file name to test file
	--successor arg			: flipbit|flipbit-neighbors|flipbit-confidences-neighbors|stochastic|stochastic-neighbors|stochastic-confidences-neighbors|cut-schedule|cut-schedule-neighbors|cut-schedule-confidences-neighbors
	--unique-iter arg		: unique iteration ID (num-test-iters needs to be 1)
	--verbose arg			: turn on verbose output if true

Notes:
* The first three arguments are required. They are the input directory, output directory and time bound.
* Can use multiple --infer and --learn options in any order to define a schedule. Must come after the mandatory arguments.
```

### Postprocessing Module

The postprocessing modules are written in MATLAB. Before running them, make sure to set up LIBSVM.

The main visualization postprocessing module is `$ROOT$/postprocess/visualize_results.m`:

```
function visualize_results( rawDir, preprocessedDir, resultsDir, outputDir, label2color, timeRange, foldRange, searchTypes, splitsName, allData )
%VISUALIZE_RESULTS Visualize results folder.
%
%	rawDir:             folder path containing original images and annotations
%                           e.g. 'DataRaw/SomeDataset'
%	preprocessedDir:	folder path containing preprocessed data
%                           e.g. 'DataPreprocessed/SomeDataset'
%	resultsDir:         folder path containing HC-Search results
%                           e.g. 'Results/SomeExperiment'
%	outputDir:          folder path to output visualization
%                           e.g. 'ResultsPostprocessed/SomeExperiment'
%   label2color:        mapping from labels to colors (use containers.Map)
%   timeRange:          range of time bound
%   foldRange:          range of folds
%   searchTypes:        list of search types 1 = HC, 2 = HL, 3 = LC, 4 = LL
%   splitsName:         (optional) alternate name to splits folder
%	allData:            (optional) data structure containing all preprocessed data
```

## Technical Documentation

Technical documentation (i.e. how to use the C++ API) is available in the `$ROOT$/doc` directory.

If not available, then use Doxygen (doxygen.org) to generate the documentation from source code. Run the command `doxygen` from the command line in the directory containing the `Doxyfile`.
