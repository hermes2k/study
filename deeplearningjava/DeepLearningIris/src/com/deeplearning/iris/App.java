package com.deeplearning.iris;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer.Builder;
import java.util.Random;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App {
	public static void main(String[] args) {

		int numOutput = 3;
		int numSamples = 150;
		int batchSize = 150;
		int splitTrainNum = (int) (batchSize * .8);
		int seed = 123;
		DataSetIterator iter = new IrisDataSetIterator(batchSize, numSamples);
		DataSet next = iter.next();
		next.shuffle();
		next.normalize();

		SplitTestAndTrain testAndTrain = next.splitTestAndTrain(splitTrainNum, new Random(seed));
		DataSet train = testAndTrain.getTrain();
		DataSet test = testAndTrain.getTest();

		// configuration related information (using builder pattern)
		NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
		// of course we have to define the number of iterations
		builder.iterations(40000);
		// learning rate
		builder.learningRate(0.01);
		// to get the same results
		builder.seed(123);
		// stochastic gradient descent as the optimization algorithm
		// it is faster but not as accurate as GRADIENT_DESCENT
		builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
		// bias nodes will be 0s at the beginning
		builder.biasInit(0);
	
		// HIDDEN LAYER !!!
		// dense means it is fully connected: every node with every node in the
		// next layer
		DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
		// there are 4 neurons in the previous (input layer) so we have 4 features
		hiddenLayerBuilder.nIn(4);
		// number of outgoing connections, nOut defines number of neurons in
		// this layer
		hiddenLayerBuilder.nOut(5);
		// we can choose whatever activation function we prefer
		hiddenLayerBuilder.activation(Activation.SIGMOID);
		// randomly initialize the weights between [0,1]
		hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		hiddenLayerBuilder.dist(new UniformDistribution(0, 1));

		// HIDDEN LAYER !!!
		// dense means it is fully connected: every node with every node in the
		// next layer
		DenseLayer.Builder hiddenLayerBuilder2 = new DenseLayer.Builder();
		// there are 4 neurons in the previous layer
		hiddenLayerBuilder2.nIn(5);
		// number of outgoing connections, nOut defines number of neurons in
		// this layer
		hiddenLayerBuilder2.nOut(5);
		// we can choose whatever activation function we prefer
		hiddenLayerBuilder2.activation(Activation.SIGMOID);
		// randomly initialize the weights between [0,1]
		hiddenLayerBuilder2.weightInit(WeightInit.DISTRIBUTION);
		hiddenLayerBuilder2.dist(new UniformDistribution(0, 1));

		// OUTPUT LAYER !!!
		// L(w) loss function is the negative log likelihood
		Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
		// must be the same amount as neurons in the layer before
		outputLayerBuilder.nIn(5);
		// two neurons in the output layer because we have 3 classes (100, 010 and 001)
		outputLayerBuilder.nOut(3);
		// for classification purposes we usually use softmax function
		outputLayerBuilder.activation(Activation.SOFTMAX);
		outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
		// again the weights are initialized at random between [0,1]
		outputLayerBuilder.dist(new UniformDistribution(0, 1));

		// so far we have defined the layers. Now we can build the layers !!!
		ListBuilder listBuilder = builder.list();
		// we know there must be an input layer so we just have to deal with
		// hidden layers + output layer
		listBuilder.layer(0, hiddenLayerBuilder.build());
		listBuilder.layer(1, hiddenLayerBuilder2.build());
		listBuilder.layer(2, outputLayerBuilder.build());
		// no pretrain phase for this network
		listBuilder.pretrain(false);

		// seems to be mandatory
		//listBuilder.backprop(true);

		// build and initialize the network: checks if the configuration is fine
		MultiLayerConfiguration networkConfiguration = listBuilder.build();
		MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(networkConfiguration);
		neuralNetwork.init();

		// print the error on every 100 iterations
		neuralNetwork.setListeners(new ScoreIterationListener(100));

		//learning on the training set exclusively
		neuralNetwork.fit(train);

		//we evaluate the model on the test set
		Evaluation evaluation = new Evaluation(numOutput);
		evaluation.eval(test.getLabels(), neuralNetwork.output(test.getFeatureMatrix(), Layer.TrainingMode.TEST));
		System.out.println(evaluation.stats());
	}
}
