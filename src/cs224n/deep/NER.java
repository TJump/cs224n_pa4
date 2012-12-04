package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    
    public static void main(String[] args) throws IOException {
		if (args.length < 2) {
		    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
		    return;
		}	    		
	
		// this reads in the train and test datasets
		List<Datum> trainData = FeatureFactory.readTrainData(args[0]);
		List<Datum> testData = FeatureFactory.readTestData(args[1]);	
		
		//	read the train and test data
		FeatureFactory.initializeVocab("data/vocab.txt");
		FeatureFactory.readWordVectors("data/wordVectors.txt");
//		FeatureFactory.saveTopWords(trainData, "top10k.txt", 10000);
		
		// initialize model 
		int windowSize = 7;
		int hiddenSize = 100;
		double learningRate = 0.001;
		double regularization = 3;
		
		int iter = 40;
//		WindowModel model = new WindowModel(trainData, testData, windowSize, hiddenSize, learningRate, regularization, iter);
		WindowModel model = null;
//		int[] windowSizeArray = new int[]{3, 5, 7, 9};
//		double[] learningRateArray = new double[]{0.003, 0.005, 0.01};
		int[] hiddenSizeArray = new int[]{100};
		
//		model = new WindowModel(trainData, testData, windowSize, hiddenSize, learningRate, regularization, iter);
//		model.loadParametersFromFile();
//		model.saveL("learnedL.csv");
		
		for(int h : hiddenSizeArray){
			hiddenSize = h;
			model = new WindowModel(trainData, testData, windowSize, hiddenSize, learningRate, regularization, iter);			
//			model.saveL("originalL");
			model.loadParametersFromFile();
//			model.saveL("learnedL");
			model.test(true);
//			model.train(false);
			
//			model.test();
			
		}
		
    }
}