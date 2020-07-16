1)Requirements:
--Visual Studio 2019
--Cuda - version 10.1, Cudnn - version 7.6.5, Tensorrt version 7.0.0.11
--opencv - version 4.1.2
--cmake - version 3.18.0-rc2


2)The project structure:

--folder research: Python notebook and scripts for dataset loading, model formation, training, testing, saving model and conversion to .pb and .uff file.
	--folder model: will contain .h5 model file
	--folder findings: Graphs developed during training for monitoring accuracy and loss
	--folder inference: 
		--folder src: CmakeLists.txt, c++ source file for generating engine from uff and inference on test_data
		--folder lib: Contains all the libaries for linking to running the project
		--folder include: contains the header files required by the project
		--folder test_data: Testing dataset required during the time of inference
		--label.txt : file containing class names of cifar10 dataset required during inference
    
    
3)Steps for running the project:

	1)go to folder research and run "training" notebook for dataset loading, model formation, training, testing(calculating accuracy and runtime), saving graphs, 
		and saving model as .h5 file
	2)go to folder research and run "model_h5_to_uff_conversion.py" for generating .uff model file from .h5 file
	3)go to folder inference/src:
		--open cmake-gui from cmake-zip-path-installation/bin folder:
			--set src folder path in source code path in gui
			--set build libaries path as src folder path/build in gui
			--click on configure and give permission to make build directory, specify the generator as visual studio 16 2019 and then click on generate.
		
		--open build folder and click on "inference.sln" which will open in visual studio
		--in visual studio on left pane right click on project name "inference" and click on set as startup project
		--click on project in top menu and go to properties:
			--in properties: 
				--go to VC++ Directories and click on include Directories, click on edit and add the "include" directories path of 
				"Nvidia GPU Cuda Toolkit" and "opencv"
				--go to linker/input: click on additional dependencies, edit and paste path "pathto NVIDIA GPU Computing Toolkit\CUDA\v10.1\lib\x64\*.lib"
				which is the path of nvidia gpu toolkit library files.
				--click ok ok untill ypu exit dialogue boxes.
		--click on local windows debugger in the top menu pane in visual studio to build and run the project solution.
