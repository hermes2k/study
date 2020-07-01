package com.deeplearning.recurrent;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
import java.net.URL;

public class App {

    //first we need a dataset as usual: 25k training examples with labels
	//movie reviews with two outputs: [0,1] for negative, [1,0] for positive review
    public static final String DATASET_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
    //location where to download and extract this huge dataset
    public static final String DATASET_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");
    //GoogleNewsVector contains the vectors associated with every word: this is how we can compare words and
    //make the sentiment analysis in the end
    public static final String WORD_VECTORS_PATH = "C:\\Users\\User\\Downloads\\GoogleNews-vectors-negative300.bin.gz";

    public static void main(String[] args) throws Exception {     

        //first of all let's download the training/test dataset from the web
        downloadData();

        //number of examples in each minibatch
        int batchSize = 64; 
        //size of the word vectors - 300 in the Google News model
        int vectorSize = 300;   
        //epoch = fill pass of training data
        int numOfEpochs = 1; 
        //truncate reviews with length (number of words) greater than this
        int truncateLength = 256;  
        final int seed = 0;   

        //we disable periodic garbage collection calls: 10.000 milliseconds
        Nd4j.getMemoryManager().setAutoGcWindow(10000);  //https://deeplearning4j.org/workspaces

        //Set up network configuration
        MultiLayerConfiguration networkConfiguration = new NeuralNetConfiguration.Builder()
            .seed(seed)
            //documentation discuss how to update the weights (and how to handle learning rate and momentum)
            .updater(Updater.ADAM) 
            //use L2 regularization to avoid overfitting
            .regularization(true).l2(1e-5)
            //Xavier initialization helps to avoid vanishing gradients problem
            .weightInit(WeightInit.XAVIER)
            //maximizing the gradient helps to avoid exploding gradient problem
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .learningRate(2e-2)
            //going to be a bit slower but uses less memory ... tradeoff
            //trainingWorkspaceMode = forward propagation + backpropagation ... inferenceWorkspaceMode = just forward propagation
            .trainingWorkspaceMode(WorkspaceMode.SEPARATE).inferenceWorkspaceMode(WorkspaceMode.SEPARATE)  
            .list()
            //this how we define the recurrent neural networks architecture: we use Long-Short Term Memory (LSTM)
            .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
                .activation(Activation.TANH).build())
            //of course we have the output layer with softmax activation function
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
            //MCXENT loss function: multi-class cross entropy loss function
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(2).build())
            .pretrain(false).backprop(true).build();

        //let's create and initialize the recurrent neural network
        MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(networkConfiguration);
        neuralNetwork.init();
        neuralNetwork.setListeners(new ScoreIterationListener(1));

        //DataSetIterators for training and testing
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        SentimentExampleIterator trainingDataset = new SentimentExampleIterator(DATASET_PATH, wordVectors, batchSize, truncateLength, true);
        SentimentExampleIterator testDataset = new SentimentExampleIterator(DATASET_PATH, wordVectors, batchSize, truncateLength, false);

        //let's train the neural network
        for (int i = 0; i < numOfEpochs; i++) {
            neuralNetwork.fit(trainingDataset);
            trainingDataset.reset();
            
            Evaluation evaluation = neuralNetwork.evaluate(testDataset);
            System.out.println(evaluation.stats());
        }

        //let's create a review and test our recurrent neural network
        String firstPositiveReview = "To be honest, we liked it! This movie was great, and I suggest that you go see it before you judge";
        //algorithms (machine learning, AI) are dealing with numerical values not strings
        INDArray features = testDataset.loadFeaturesFromString(firstPositiveReview, truncateLength);
        System.out.println("Feature representation of the sentence: " + features);
        //prediction made by the neural network
        INDArray networkOutput = neuralNetwork.output(features);
        int numOfOutputs = networkOutput.size(2);
        //transform the prediction into probabilities
        INDArray sentimentProbabilities = networkOutput.get(NDArrayIndex.point(0), NDArrayIndex.all(), NDArrayIndex.point(numOfOutputs - 1));

        System.out.println("\n\nOur review: " + firstPositiveReview);
        System.out.println("Result prediction: " + networkOutput);
        //[1,0] means positive review
        System.out.println("p(positive): " + sentimentProbabilities.getDouble(0));
        //[0,1] means negative review
        System.out.println("p(negative): " + sentimentProbabilities.getDouble(1));
    }

    public static void downloadData() throws Exception {
        //Create directory if required
        File directory = new File(DATASET_PATH);
        if(!directory.exists()) directory.mkdir();

        //Download file:
        String archizePath = DATASET_PATH + "aclImdb_v1.tar.gz";
        File archiveFile = new File(archizePath);
        String extractedPath = DATASET_PATH + "aclImdb";
        File extractedFile = new File(extractedPath);

        if( !archiveFile.exists() ){
            System.out.println("Starting data download (80MB)...");
            FileUtils.copyURLToFile(new URL(DATASET_URL), archiveFile);
            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
            //Extract tar.gz file to output directory
            DataUtilities.extractTarGz(archizePath, DATASET_PATH);
        } else {
            //Assume if archive (.tar.gz) exists, then data has already been extracted
            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
            if( !extractedFile.exists()){
            	//Extract tar.gz file to output directory
            	DataUtilities.extractTarGz(archizePath, DATASET_PATH);
            } else {
            	System.out.println("Data (extracted) already exists at " + extractedFile.getAbsolutePath());
            }
        }
    }
}