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
		
		// initialize model 
		WindowModel model = new WindowModel(trainData, testData, 5, 100, 0.001, 0.0001, 15);
		
		int[] windowSize = new int[]{3, 5, 7, 9};
		for(int w : windowSize){
			model.setWindowSize(w);
			model.train();
			model.test();
		}
    }
}