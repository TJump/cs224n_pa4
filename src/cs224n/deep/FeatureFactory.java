package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import org.ejml.ops.MatrixIO;
import org.ejml.simple.*;


public class FeatureFactory {


	private FeatureFactory() {

	}

	 
	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}
	
	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}
	
	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}

		return data;
	}
 
 
	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		if (allVecs!=null) return allVecs;
		int n = 50;
		allVecs = new SimpleMatrix(n, wordToNum.size());
		BufferedReader in = new BufferedReader(new FileReader(vecFilename));
		int nCol = 0;
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			line = line.trim();
			String[] values = line.split(" ");
			int r = 0;
			for(String value : values){
				double d = Double.parseDouble(value);
				allVecs.set(r++, nCol, d);
			}
			nCol++;
		}
		return allVecs;
//		return new SimpleMatrix(MatrixIO.loadCSV(vecFilename, wordToNum.size(), 50)).transpose();
	}
	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static int getIndex(String s){
		if(wordToNum.containsKey(s)) return wordToNum.get(s);
		else return 0;
	}
	public static void initializeVocab(String vocabFilename) throws IOException {
		int i=0;
		BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			line = line.trim();
			if(!wordToNum.containsKey(line)){
				wordToNum.put(line, i);
				numToWord.put(i, line);
				i++;
			}
		}
		
//		return wordToNum;
	}
 








}
