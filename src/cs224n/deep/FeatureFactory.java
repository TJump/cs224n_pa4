package cs224n.deep;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import org.ejml.ops.MatrixIO;
import org.ejml.simple.*;


public class FeatureFactory {
	
	final static String START_WORD = "<s>";
	final static String STOP_WORD = "</s>";
	final static int N_VECTOR_SIZE = 50;
	private FeatureFactory() {}
	 
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
	
	
	public static void saveTopWords(List<Datum> train, String fileName, int topK) throws FileNotFoundException, IOException{
		Map<Integer, Integer> wordCount = new HashMap<Integer, Integer>();	
		for(Datum d : train){
			int index = getIndex(d.word);
			if(wordCount.containsKey(index)){
				wordCount.put(index, wordCount.get(index) + 1);
			} else {
				wordCount.put(index, 1);
			}
		}
		
		// http://stackoverflow.com/questions/109383/how-to-sort-a-mapkey-value-on-the-values-in-java
		ValueComparator bvc =  new ValueComparator(wordCount);
		TreeMap<Integer, Integer> sorted_map = new TreeMap<Integer, Integer>(bvc);
		sorted_map.putAll(wordCount);
		
		File f = new File(fileName);
		BufferedWriter bw = null;
		bw = new BufferedWriter(new FileWriter(f));
		int c = 1;
		for(Integer s : sorted_map.keySet()){
			if(c>topK) break;
			if(!wordCount.containsKey(s))
				System.out.println(s);
			System.out.println(numToWord.get(s));
			bw.append(s + "\n");
			c++;
		}
		bw.close();
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
		
		allVecs = new SimpleMatrix(N_VECTOR_SIZE, wordToNum.size());
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
		in.close();
		return allVecs;
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
		in.close();
	}
 
}

class ValueComparator implements Comparator<Integer> {

    Map<Integer, Integer> base;
    public ValueComparator(Map<Integer, Integer> wordCount) {
        this.base = wordCount;
    }

    // Note: this comparator imposes orderings that are inconsistent with equals.    
    public int compare(Integer a, Integer b) {
        if (base.get(a) >= base.get(b)) {
            return -1;
        } else {
            return 1;
        } // returning 0 would merge keys
    }
}
