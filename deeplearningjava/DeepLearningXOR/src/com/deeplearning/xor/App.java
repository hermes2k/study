package com.deeplearning.xor;

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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class App {
    public static void main(String[] args) {

        //the input has 2 dimensions: x and y (both logical values so 0 or 1
    	//it is like a matrix: we have to define rowIndex and colIndex accordingly
        INDArray input = Nd4j.zeros(4, 2);
        INDArray labels = Nd4j.zeros(4, 2);

        //have to define the x and y values one by one (by indexes)
        input.putScalar(new int[]{0, 0}, 0);
        input.putScalar(new int[]{0, 1}, 0);
        input.putScalar(new int[]{1, 0}, 1);
        input.putScalar(new int[]{1, 1}, 0);
        input.putScalar(new int[]{2, 0}, 0);
        input.putScalar(new int[]{2, 1}, 1);
        input.putScalar(new int[]{3, 0}, 1);
        input.putScalar(new int[]{3, 1}, 1);
        
        //we have two classes logical 0 (1 0) and logical 1 (0 1)
        //two lines --> represent one class !!!
        labels.putScalar(new int[]{0, 0}, 1);
        labels.putScalar(new int[]{0, 1}, 0);
        labels.putScalar(new int[]{1, 0}, 0);
        labels.putScalar(new int[]{1, 1}, 1);
        labels.putScalar(new int[]{2, 0}, 0);
        labels.putScalar(new int[]{2, 1}, 1);        
        labels.putScalar(new int[]{3, 0}, 1);
        labels.putScalar(new int[]{3, 1}, 0);

        //the dataset is the combination of inputs and labels
        DataSet dataset = new DataSet(input, labels);     

        //configuration related information (using builder pattern)
        NeuralNetConfiguration.Builder builder = new NeuralNetConfiguration.Builder();
        //of course we have to define the number of iterations
        builder.iterations(10000);
        //learning rate
        builder.learningRate(0.1);
        //to get the same results
        builder.seed(123);
        //dataset is too small to use dropout regularization
        builder.useDropConnect(false);
        //stochastic gradient descent as the optimization algorithm
        //it is faster but not as accurate as GRADIENT_DESCENT
        builder.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT);
        //bias nodes will be 0s at the beginning
        builder.biasInit(0);
        //minibatch is useful BUT not with such a small dataset !!!
        builder.miniBatch(false);     

        //HIDDEN LAYER !!!
        //dense means it is fully connected: every node with every node in the next layer
        DenseLayer.Builder hiddenLayerBuilder = new DenseLayer.Builder();
        //there is 2 neurons in the previous (input layer)
        hiddenLayerBuilder.nIn(2);
        //number of outgoing connections, nOut defines number of neurons in this layer
        hiddenLayerBuilder.nOut(4);
        //we can choose whatever activation function we prefer
        hiddenLayerBuilder.activation(Activation.RELU);
        //randomly initialize the weights between [0,1]
        hiddenLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        hiddenLayerBuilder.dist(new UniformDistribution(0, 1));   
           
        //HIDDEN LAYER !!!
        //dense means it is fully connected: every node with every node in the next layer
        DenseLayer.Builder hiddenLayerBuilder2 = new DenseLayer.Builder();
        //there is 2 neurons in the previous layer
        hiddenLayerBuilder2.nIn(4);
        //number of outgoing connections, nOut defines number of neurons in this layer
        hiddenLayerBuilder2.nOut(4);
        //we can choose whatever activation function we prefer
        hiddenLayerBuilder2.activation(Activation.RELU);
        //randomly initialize the weights between [0,1]
        hiddenLayerBuilder2.weightInit(WeightInit.DISTRIBUTION);
        hiddenLayerBuilder2.dist(new UniformDistribution(0, 1));

        //OUTPUT LAYER !!!
        //L(w) loss function is the negative log likelihood
        Builder outputLayerBuilder = new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD);
        // must be the same amount as neurons in the layer before
        outputLayerBuilder.nIn(4);
        // two neurons in the output layer because we have 2 classes (0 and 1)
        outputLayerBuilder.nOut(2);
        //for classification purposes we usually use softmax function
        outputLayerBuilder.activation(Activation.SOFTMAX);
        outputLayerBuilder.weightInit(WeightInit.DISTRIBUTION);
        //again the weights are initialized at random between [0,1]
        outputLayerBuilder.dist(new UniformDistribution(0, 1));
        
        //so far we have defined the layers. Now we can build the layers !!!
        ListBuilder listBuilder = builder.list();
        //we know there must be an input layer so we just have to deal with hidden layers + output layer
        listBuilder.layer(0, hiddenLayerBuilder.build());
        listBuilder.layer(1, hiddenLayerBuilder2.build());
        listBuilder.layer(2, outputLayerBuilder.build());
        //no pretrain phase for this network
        listBuilder.pretrain(false);

        // seems to be mandatory
        //listBuilder.backprop(true);
   
        //build and initialize the network: checks if the configuration is fine
        MultiLayerConfiguration networkConfiguration = listBuilder.build();
        MultiLayerNetwork neuralNetwork = new MultiLayerNetwork(networkConfiguration);
        neuralNetwork.init();

        //print the error on every 100 iterations
        neuralNetwork.setListeners(new ScoreIterationListener(100));

        //the actual learning
        neuralNetwork.fit(dataset);

        //create output for every training sample
        INDArray output = neuralNetwork.output(dataset.getFeatureMatrix());

        //prints statistics + how often the right output had the highest value + 2 is the number of classes in the output
        Evaluation evaluation = new Evaluation(2);
        evaluation.eval(dataset.getLabels(), output);
        System.out.println(evaluation.stats());
        
        //we can use the neural network to make new predictions
        System.out.println("NEW PREDICTION");
        System.out.println(neuralNetwork.output(input));
    }
}
