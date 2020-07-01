package com.deeplearning.convolutional;

import org.nd4j.linalg.activations.Activation;
import java.io.File;
import java.util.Random;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App {
   
    public static void main(String[] args) throws Exception {
        
    	//no RGB channels just a single channel
    	int numOfChannels = 1;
    	int width = 28;
    	int height = 28;
    	//dwe have two classes (happy and sad encoded in digits 0 and 1)
        int numOfOutput = 2; 
        //all the items
        int batchSize = 16; 
        //an epoch is defined as a full pass of the data set
        int numOfEpochs = 200;
        //the number of parameter updates in a row, for each minibatch
        int numOfIterations = 1;
        int seed = 123; 
        Random random = new Random();

        //we create a custom datasets so let's read the happy and sad images
        File trainData = new File("C:\\Users\\User\\Desktop\\UdemyDataset");
        //we will test our algorithm on test images
        File testData = new File("C:\\Users\\User\\Desktop\\UdemyTest");
        //we shuffle the images at random
        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, random);
        FileSplit test = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, random);
        //we want to set the labels (10 and 01) for the images automatically 
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, numOfChannels, labelMaker);
        recordReader.initialize(train);
        //we want to end up with a DataSetIterator (as the container for the images)
        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numOfOutput);

        //normalize the dataset [minRange=0,maxRange=1]
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.fit(dataIter);
        dataIter.setPreProcessor(scaler);

        //check the images and labels accordingly
        DataSet ds = dataIter.next();
        System.out.println(ds);
        System.out.println(dataIter.getLabels());
        
        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(numOfIterations) 
                .regularization(true).l2(0.0005)            
                .learningRate(.01)             
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                //Nesterov's momentum: keep track of the previous layer's gradient and use it as a way of updating the gradient
                .updater(Updater.NESTEROVS) 
                .list()
                //the filter (feature detector) size is 5x5
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        //black and white image so no RGB components just 1 channel
                        .nIn(numOfChannels)
                        //takes only a single step over and then down as it slides the filter across the input volume
                        .stride(1, 1)
                        //number of kernels: so the number of feature maps in the output
                        .nOut(20)
                        .activation(Activation.RELU)
                        .build())
                //pooling layer AKA subsampling layer with MAX pooling
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                		//filter size is 2x2
                        .kernelSize(2,2)
                        //it will step two values as it slides horizontally and two values when it steps down to the next row
                        .stride(2,2)
                        .build())
                //convolutional layers + pooling layers right after each other           
                .layer(2, new DenseLayer.Builder().activation(Activation.RELU)
                        .nOut(500).build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(numOfOutput)
                        .activation(Activation.SOFTMAX)
                        .build())
                //emoji dataset 28x28x1 grayscale images
                .setInputType(InputType.convolutional(28,28,1))
                .backprop(true).pretrain(false).build();   

        MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(configuration);
        neuralNetwork.init();

        neuralNetwork.setListeners(new ScoreIterationListener(1));
        
        //training the neural network
        for( int i=0; i<numOfEpochs; i++ ) {       	
            neuralNetwork.fit(dataIter);
            Evaluation evaluation = neuralNetwork.evaluate(dataIter);
            System.out.println(evaluation.stats());
            dataIter.reset();
        }
        
        //let's classify new images ... first we have to import the image        
        ParentPathLabelGenerator labelMakerTest = new ParentPathLabelGenerator();
        ImageRecordReader recordReaderTest = new ImageRecordReader(height, width, numOfChannels, labelMakerTest);
        recordReaderTest.initialize(test);
        //we want to end up with a DataSetIterator (as the container for the images)
        DataSetIterator dataIterTest = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 1, numOfOutput);

        //normalize the dataset [minRange=0,maxRange=1]
        DataNormalization scalerTest = new ImagePreProcessingScaler(0, 1);
        scalerTest.fit(dataIterTest);
        dataIterTest.setPreProcessor(scalerTest);
        
        //MAKING PREDICTION WITH THE NEURAL NETWORK [1 0] happy       [0 1] sad
        System.out.println("New prediction: " + neuralNetwork.output(dataIterTest));
    }
}
