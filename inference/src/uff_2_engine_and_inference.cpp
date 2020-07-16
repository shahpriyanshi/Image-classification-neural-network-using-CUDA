#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cuda_runtime_api.h>
#include <opencv.hpp>
#include <NvInfer.h>
#include <NvUffParser.h>
#include "utils.h"

using namespace std;
using namespace nvinfer1;
using namespace nvuffparser;


class Logger : public ILogger
{
    void log(Severity severity, const char* msg) override
    {
        cout << msg << endl;
    }
} gLogger;


int main()
{
    
    int inputHeight = 32;
    int inputWidth = 32;
    int maxBatchSize = 1;
    DataType dataType = DataType::kFLOAT;

    /* parse uff */
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetwork();
    IUffParser* parser = createUffParser();
    parser->registerInput("conv2d_1_input", DimsCHW(3, inputHeight, inputWidth), UffInputOrder::kNCHW);
    parser->registerOutput("dense_2/Softmax");
    if (!parser->parse("../../../model/image_classification_model.uff", *network, dataType))
    {
        cout << "Failed to parse UFF\n";
        builder->destroy();
        parser->destroy();
        network->destroy();
        return 1;
    }

    /* build engine */
    if (dataType == DataType::kHALF)
        builder->setHalf2Mode(true);

    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1<<20);
    ICudaEngine* engine = builder->buildCudaEngine(*network);

    if (engine == nullptr)
    {
        cout << "Could not open plan file." << endl;
        return 1;
    }
    IExecutionContext* context = engine->createExecutionContext();

    /* get the input / output dimensions */
    int inputBindingIndex, outputBindingIndex;
    inputBindingIndex = engine->getBindingIndex("conv2d_1_input");
    outputBindingIndex = engine->getBindingIndex("dense_2/Softmax");

    if (inputBindingIndex < 0)
    {
        cout << "Invalid input name." << endl;
        return 1;
    }

    if (outputBindingIndex < 0)
    {
        cout << "Invalid output name." << endl;
        return 1;
    }

    Dims inputDims, outputDims;
    inputDims = engine->getBindingDimensions(inputBindingIndex);
    outputDims = engine->getBindingDimensions(outputBindingIndex);

    /* read image, convert color, and resize */
    cout << "Preprocessing test_data..." << endl;

    float totalTime = 0.0;
    const int kBatchSize = 1;

    ifstream labelsFile("../../labels.txt");

    if (!labelsFile.is_open())
    {
        cout << "\nCould not open label file." << endl;
        return 1;
    }
    vector<string> labelMap;

    string label;
    while (getline(labelsFile, label))
    {
        labelMap.push_back(label);
    }


    for (int i = 0; i < 1000; i++) {

        cv::Mat image = cv::imread("../../test_data/test"+to_string(i)+".jpg", cv::IMREAD_COLOR);


        if (!image.data)
        {
            cout << "Could not read image from file." << endl;
            return 1;
        }

        cv::cvtColor(image, image, cv::COLOR_BGR2RGB, 3);
        cv::resize(image, image, cv::Size(inputWidth, inputHeight));

        /* convert from uint8+NHWC to float+NCHW */
        float* inputDataHost, * outputDataHost;
        size_t numInput, numOutput;
        numInput = numTensorElements(inputDims);
        numOutput = numTensorElements(outputDims);
        inputDataHost = (float*)malloc(numInput * sizeof(float));
        outputDataHost = (float*)malloc(numOutput * sizeof(float));
        cvImageToTensor(image, inputDataHost, inputDims);
        preprocessImage(inputDataHost, inputDims);

        /* transfer to device */
        void* inputDataDevice, * outputDataDevice;
        cudaMalloc(&inputDataDevice, numInput * sizeof(float));
        cudaMalloc(&outputDataDevice, numOutput * sizeof(float));

        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);

        float elapsedTime;
        

        cudaMemcpy(inputDataDevice, inputDataHost, numInput * sizeof(float), cudaMemcpyHostToDevice);

        void* bindings[2];
        bindings[inputBindingIndex] = (void*)inputDataDevice;
        bindings[outputBindingIndex] = (void*)outputDataDevice;

        cudaEventRecord(start);
        context->execute(kBatchSize, bindings);
        cudaEventRecord(end);

        /* transfer output back to host */
        cudaMemcpy(outputDataHost, outputDataDevice, numOutput * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsedTime, start, end);
        totalTime += elapsedTime;

        /* parse output */
        vector<size_t> sortedIndices = argsort(outputDataHost, outputDims);

        cout << "\nThe predicted class index of image: test" + to_string(i) + ".jpg is: ";

        cout << sortedIndices[0] << " ";
        
        cout << "\nWhich corresponds to class label: ";
        cout << endl << labelMap[sortedIndices[0]];
        cout << endl;

        free(inputDataHost);
        free(outputDataHost);
        cudaFree(inputDataDevice);
        cudaFree(outputDataDevice);

    }

    cout << "Inference batch size " << kBatchSize << " runs in " << totalTime << "ms" << endl;

    /* clean up */

    engine->destroy();
    context->destroy();
    

    /* break down */
    builder->destroy();
    parser->destroy();
    network->destroy();

    return 0;
}