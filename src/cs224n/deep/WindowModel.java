package cs224n.deep;
import java.util.*;

import org.ejml.simple.*;


public class WindowModel {
	
	// Gradient Check
	final static boolean GRADCHECK = !true;
	final static double EPSILON = 1e-4;
	final static double THRESHOLD = 1e-7;
	
	final static int MAX_TRAIN_SIZE = 300000;
	final static int N_VECTOR_SIZE = 50;
	
	// Default hyperparameters
	final static double DEFAULT_REGULARIZATION_WEIGHT = 0.001;
	final static double DEFAULT_LEARNING_RATE = 0.001;
	final static int DEFAULT_WINDOW_SIZE = 5;
	final static int DEFAULT_HIDDEN_SIZE = 100;
	final static int DEFAULT_NUM_ITERATIONS = 2;
	
	// Parameters in the cost function to be optimized
	private SimpleMatrix L, W, U, b1;
	private double b2;
	
	// Hyperparameters to be tuned
	private double regularizationWeight;
	private double learningRate;
	private int numIterations;
	private int windowSize, hiddenSize;

	// default constructor
	public WindowModel(){
		this(DEFAULT_WINDOW_SIZE, DEFAULT_HIDDEN_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_REGULARIZATION_WEIGHT, DEFAULT_NUM_ITERATIONS);
	}
	
	public WindowModel(int _windowSize, int _hiddenSize, double _learningRate, double _regWeight, int _numIter){
		windowSize = _windowSize;
		hiddenSize = _hiddenSize;
		learningRate = _learningRate;
		numIterations = _numIter;
		regularizationWeight = _regWeight;
		
		// Dimension(L) = n x V
		L = FeatureFactory.allVecs;
	}		

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		int fanIn = N_VECTOR_SIZE * windowSize;
		int fanOut = hiddenSize;
		double epsilonInit = Math.sqrt(6.0 / (fanIn + fanOut));
		
		// Dimension(W) = H x C*n
		W = SimpleMatrix.random(hiddenSize, windowSize * N_VECTOR_SIZE, -epsilonInit, epsilonInit, new Random());
		U = SimpleMatrix.random(hiddenSize, 1, -epsilonInit, epsilonInit, new Random());
		b1 = SimpleMatrix.random(hiddenSize, 1, -epsilonInit, epsilonInit, new Random());
		b2 = new Random().nextDouble() * epsilonInit;
	}
	
	private double getH(SimpleMatrix x){
		
		SimpleMatrix z = W.mult(x).plus(b1);
		SimpleMatrix a = getTanhMatrix(z);
		SimpleMatrix u = U.transpose().mult(a);
		return getSigmoid(u.get(0, 0) + b2);
	}

	// tanh
	private static SimpleMatrix getTanhMatrix(SimpleMatrix x){
		SimpleMatrix y = new SimpleMatrix(x);
		for(int i=0;i<x.numRows();i++)
			for(int j=0;j<x.numCols();j++)
				y.set(i, j, Math.tanh(y.get(i, j)));
		return y;
	}
	
	private static SimpleMatrix getTanhDerivativeMatrix(SimpleMatrix x){
		SimpleMatrix y = new SimpleMatrix(x);
		for(int i=0;i<x.numRows();i++)
			for(int j=0;j<x.numCols();j++){
				double t = Math.tanh(y.get(i, j));
				y.set(i, j, 1-Math.pow(t, 2));
			}
		return y;
	}
	
	public static double getTanhDerivative(double x){
		return 1 - Math.pow(Math.tanh(x), 2);
	}
	
	public static double getSigmoid(double x){
		return 1/(1+Math.exp(-x));
	}
	
	
	private SimpleMatrix getStartOrEnd(boolean startOrEnd){
		return getWordVector(startOrEnd ? FeatureFactory.START_WORD : FeatureFactory.STOP_WORD);
	}
		
	private SimpleMatrix getWordVector(String s){
		// index of unkown word = 0
		int index = FeatureFactory.wordToNum.containsKey(s) ? FeatureFactory.wordToNum.get(s) : 0; 
		return L.extractVector(false, index);
	}
	
	int trainingSizeM = -1;
	
	private IntTuple extractTrainingExample(int j, List<Datum> _trainData, SimpleMatrix feature, int sentenceIndex, List<Integer> wordIndexList){
		int halfWindowSize = windowSize/2;
		Datum d = _trainData.get(j);
		String centerWord = d.word;
		String lable = d.label;
		int y = lable.equals("O") ? 0 : 1;
		int numStartTagTofill = Math.max(0, halfWindowSize - sentenceIndex);
		
		for(int k=0; k<numStartTagTofill; k++){
			wordIndexList.add(FeatureFactory.wordToNum.get(FeatureFactory.START_WORD ));
			feature.insertIntoThis(k * N_VECTOR_SIZE, 0, getStartOrEnd(true));
		}
		
		int numVectorTofill = halfWindowSize - numStartTagTofill;
		for(int k=0;k<numVectorTofill;k++){
			Datum d_ = _trainData.get(j-numVectorTofill+k);
			String word_ = d_.word;
			wordIndexList.add(FeatureFactory.getIndex(word_));
			SimpleMatrix theVector = getWordVector(word_);
			feature.insertIntoThis((k + numStartTagTofill) * N_VECTOR_SIZE, 0, theVector);
		}
		
		// over x... end of window size
		for(int k=0;k<=halfWindowSize;k++){
			
			Datum d_ = _trainData.get(j+k);
			String word_ = d_.word;
			wordIndexList.add(FeatureFactory.getIndex(word_));
			SimpleMatrix theVector = getWordVector(word_);
			feature.insertIntoThis((k + halfWindowSize) * N_VECTOR_SIZE, 0, theVector);
			
			if(j+k == _trainData.size()-1 || word_.equals(".")){
				while(k<halfWindowSize){
					wordIndexList.add(FeatureFactory.wordToNum.get(FeatureFactory.STOP_WORD));
					feature.insertIntoThis((k + halfWindowSize) * N_VECTOR_SIZE, 0, getStartOrEnd(false));
					k++;
				}
				break;
			}
		}
		
		if(centerWord.equals(".")){
			sentenceIndex = 0;
		} else {
			sentenceIndex++;
		}
		return new IntTuple(y, sentenceIndex);
	}
	
	
	public void train(List<Datum> _trainData){
		int trainSize = _trainData.size();
		System.out.println("===================");
		System.out.println("training started...");
		System.out.printf("\titerations = %d, training size: full = %d, limit = %d\n", numIterations, trainSize, MAX_TRAIN_SIZE);
		System.out.printf("\tlearning rate = %f, reg weight = %f\n", learningRate, regularizationWeight);
		long startTime = System.currentTimeMillis();
		
		trainingSizeM = _trainData.size();
		
		for(int i=1;i<=numIterations;i++){
			
			System.out.println("\tepoch # " + i);
			double costSum = 0;
			Integer sentenceIndex = new Integer(0);
			
			for(int j=0; j<_trainData.size(); j++){
				
				if(j%50000==0) System.out.println("\t\t#i = " + j);
				if(MAX_TRAIN_SIZE > 0 && j > MAX_TRAIN_SIZE) break;

				//-------------------------------------------------------
				// Getting a feature and its prediction
				LinkedList<Integer> wordIndexList = new LinkedList<Integer>();
				SimpleMatrix feature = new SimpleMatrix(windowSize * N_VECTOR_SIZE, 1);
				IntTuple tuple = extractTrainingExample(j, _trainData, feature, sentenceIndex, wordIndexList);
				int y = tuple.getFirst();
				sentenceIndex = tuple.getSecond();

				//-------------------------------------------------------
				// Gradient Check
				if(GRADCHECK){
					boolean check = gradientCheck(feature, y); 
					System.out.println("gradient check " + (check? "passed!" : "failed!") + "\n");
				}
				
				//-------------------------------------------------------
				// SGD
				double updatedb2;
				SimpleMatrix updatedU, updatedW, updatedb1;
				
				// shared vars
				SimpleMatrix z = getTanhMatrix(W.mult(feature).plus(b1));
				SimpleMatrix dz = getTanhDerivativeMatrix(z);
				Double h = getH(feature);	
				
				costSum += getCostNonRegTerm(feature, y, h);
				
				double factor = -y/h + (1-y)/(1-h);
				h *= 1-h;
				Double factor_h = factor * h;
				
				updatedU = U.minus(getD_J_U(feature, y, z, factor_h).scale(learningRate));
				updatedb2 =  b2 - learningRate * getD_J_b2(feature, y, factor_h);
				updatedW = W.minus(getD_J_W(feature, y, dz, factor_h).scale(learningRate));
				updatedb1 = b1.minus(getD_J_b1(feature, y, dz, factor_h).scale(learningRate));
				
				// update L
				List<SimpleMatrix> updatedLlist = new ArrayList<SimpleMatrix>(windowSize);
				SimpleMatrix d_j_l = getD_J_L(feature, y, dz, factor_h).scale(learningRate);
				for(int k=0; k<windowSize; k++){
					int Lindex = wordIndexList.get(k);
					int startIndex = N_VECTOR_SIZE*k;
					int endIndex = N_VECTOR_SIZE*(k+1);	
					SimpleMatrix toUpdate = d_j_l.extractMatrix(startIndex, endIndex, 0, 1);
					
					// L(:, Lindex) - learningRate * d_k
					updatedLlist.add(L.extractVector(false, Lindex).minus(toUpdate));
				}
				
				// Now, let's update the rest of parameters
				U = updatedU;
				b1 = updatedb1;
				W = updatedW;
				b2 = updatedb2;
				for(int k=0; k<windowSize; k++){
					L.insertIntoThis(0, wordIndexList.get(k), updatedLlist.get(k));
				}
			}
			
			double totalCost_J = (costSum + getCostRegTerm())/(double)trainSize;
			System.out.println("\t\ttotal cost = " + totalCost_J);
		}
		long endTime = System.currentTimeMillis();
		System.out.println("training took " + (endTime - startTime)/1000 + " sec.");
	}
	
	enum PARAMETERS {U, W, B1, B2, L};
	private SimpleMatrix getApproximateGradient(SimpleMatrix feature, int y, PARAMETERS params){
		SimpleMatrix result = null;
		
		if(params == PARAMETERS.W){
			// W: H x c*n
			result = new SimpleMatrix(W);
			for(int i=0;i<hiddenSize;i++){
				for(int j=0;j<windowSize*N_VECTOR_SIZE;j++){
					double t = W.get(i, j);
					
					W.set(i, j, t + EPSILON);
					double d_j_W_plus = getCostJ(feature, y);
					
					W.set(i, j, t - EPSILON);
					double d_j_W_minus = getCostJ(feature, y);
					
					// convert back
					W.set(i, j, t);
					result.set(i, j, 0.5*(d_j_W_plus - d_j_W_minus)/EPSILON);
				}
			}
		}
		else if(params == PARAMETERS.U){
			// U: H x 1
			result = new SimpleMatrix(U);
			for(int i=0;i<hiddenSize;i++){
				double t = U.get(i, 0);
				
				U.set(i, 0, t + EPSILON);
				double d_j_U_plus = getCostJ(feature, y);
				
				U.set(i, 0, t - EPSILON);
				double d_j_U_minus = getCostJ(feature, y);
				
				// convert back
				U.set(i, 0, t);
				result.set(i, 0, 0.5*(d_j_U_plus - d_j_U_minus)/EPSILON);
			}
		}
		else if(params == PARAMETERS.B1){
			// b1: H x 1
			result = new SimpleMatrix(b1);
			for(int i=0;i<hiddenSize;i++){
				double t = b1.get(i, 0);
				
				b1.set(i, 0, t + EPSILON);
				double d_j_b1_plus = getCostJ(feature, y);
				
				b1.set(i, 0, t - EPSILON);
				double d_j_b1_minus = getCostJ(feature, y);
				
				// convert back
				b1.set(i, 0, t);
				result.set(i, 0, 0.5*(d_j_b1_plus - d_j_b1_minus)/EPSILON);
			}
		}
		else if(params == PARAMETERS.B2){
			// 1x1 matrix for a single value
			result = new SimpleMatrix(1, 1);
			double t = b2;
			b2 = t + EPSILON;
			double j1 = getCostJ(feature, y);
			b2 = t - EPSILON;
			double j2 = getCostJ(feature, y);
			result.set(0, 0, 0.5 * (j1-j2) / EPSILON);
			b2 = t;
		}
		else if(params == PARAMETERS.L){
			result = new SimpleMatrix(feature);
			
			for(int i=0;i<windowSize * N_VECTOR_SIZE;i++){
				double t = feature.get(i, 0);
				
				feature.set(i, 0, t + EPSILON);
				double d_j_x_plus = getCostJ(feature, y);
				
				feature.set(i, 0, t - EPSILON);
				double d_j_x_minus = getCostJ(feature, y);
				
				feature.set(i, 0, t);
				result.set(i, 0, 0.5*(d_j_x_plus - d_j_x_minus)/EPSILON);
			}
		}
		
		return result;
	}
	private boolean gradientCheck(SimpleMatrix feature, int y){
		System.out.println("gradient check...");
		for(PARAMETERS param : PARAMETERS.values()){
			SimpleMatrix grad = null;
			SimpleMatrix gradApprox = getApproximateGradient(feature, y, param);
			
			if(gradApprox == null)
				continue;
			
			if(param == PARAMETERS.W) grad = getD_J_W(feature, y, null, null);
			else if(param == PARAMETERS.U) grad = getD_J_U(feature, y, null, null);
			else if(param == PARAMETERS.B1) grad = getD_J_b1(feature, y, null, null);
			else if(param == PARAMETERS.L) grad = getD_J_L(feature, y, null, null);
			else if(param == PARAMETERS.B2) {
				grad = new SimpleMatrix(1, 1);
				grad.set(0, 0, getD_J_b2(feature, y, null));
			}
			
			double delta = grad.minus(gradApprox).normF();
			System.out.println("\t" + param.toString() + " delta = " + delta);
			if(delta > THRESHOLD) return false;
		}
		return true;
	}
	
	private double getCostJ(SimpleMatrix x, int y){
		double hValue = getH(x);
		double regTerm = 0.5 * regularizationWeight * (W.elementMult(W).elementSum() + U.elementMult(U).elementSum());
		return -y * Math.log(hValue) + (y-1)*Math.log(1-hValue) + regTerm;
	}
	
	private double getCostNonRegTerm(SimpleMatrix x, int y, Double h_){
		double hValue = h_ == null ? getH(x) : h_;
		return -y * Math.log(hValue) + (y-1)*Math.log(1-hValue);
	}	
	
	private double getCostRegTerm(){
		return 0.5 * regularizationWeight * (W.elementMult(W).elementSum() + U.elementMult(U).elementSum());
	}
	
	// dw(i,j) = sigmoid * (1-sigmoid) * [u(i,0) * D[tanh^2 (w(i,j) * x(j,0))] * x(j,0)]
	private SimpleMatrix getD_J_W(SimpleMatrix x, double y, SimpleMatrix dz_, Double factor_h_){
		SimpleMatrix dwReg = W.scale(regularizationWeight);
		SimpleMatrix z = dz_ == null ? getTanhDerivativeMatrix(W.mult(x).plus(b1)) : dz_;
		SimpleMatrix dw = z.elementMult(U).mult(x.transpose());
		
		double factor_h;
		if(factor_h_ == null){
			double h = getH(x);
			double factor = -y/h + (1-y)/(1-h);
			h *= 1-h;
			factor_h = factor * h;
		}
		else{
			factor_h = factor_h_;
		}
		
		return dw.scale(factor_h).plus(dwReg);
	}
	
	private SimpleMatrix getD_J_L(SimpleMatrix x, double y, SimpleMatrix dz_, Double factor_h_){
		SimpleMatrix z = dz_ == null ? getTanhDerivativeMatrix(W.mult(x).plus(b1)) : dz_;
		SimpleMatrix dl = W.transpose().mult(z.elementMult(U));
		
		double factor_h;
		if(factor_h_ == null){
			double h = getH(x);
			double factor = -y/h + (1-y)/(1-h);
			h *= 1-h;
			factor_h = factor * h;
		}
		else{
			factor_h = factor_h_;
		}
		
		return dl.scale(factor_h);
	}
	
	private double getD_J_b2(SimpleMatrix x, double y, Double factor_h_){
		
		double factor_h;
		if(factor_h_ == null){
			double h = getH(x);
			double factor = -y/h + (1-y)/(1-h);
			h *= 1-h;
			factor_h = factor * h;
		}
		else{
			factor_h = factor_h_;
		}
		
		return factor_h;
	}
	
	// sigmoid * (1-sigmoid) * u(i,0) * (1-tanh^2 (b(i,0)))
	private SimpleMatrix getD_J_b1(SimpleMatrix x, double y, SimpleMatrix dz_, Double factor_h_){
		SimpleMatrix dz = dz_ == null? getTanhDerivativeMatrix(W.mult(x).plus(b1)) : dz_;
		SimpleMatrix db = U.elementMult(dz);
		
		double factor_h;
		if(factor_h_ == null){
			double h = getH(x);
			double factor = -y/h + (1-y)/(1-h);
			h *= 1-h;
			factor_h = factor * h;
		}
		else{
			factor_h = factor_h_;
		}
		
		return db.scale(factor_h);
	}
	
	// return sigmoid * (1-sigmoid) * tanh(wx+b1)
	private SimpleMatrix getD_J_U(SimpleMatrix x, double y, SimpleMatrix z_, Double factor_h_){
		
		SimpleMatrix z = z_ == null? getTanhMatrix(W.mult(x).plus(b1)) : z_;
		SimpleMatrix zReg = U.scale(regularizationWeight);
		double factor_h;
		if(factor_h_ == null){
			double h = getH(x);
			double factor = -y/h + (1-y)/(1-h);
			h *= 1-h;
			factor_h = factor * h;
		}
		else{
			factor_h = factor_h_;
		}
		return z.scale(factor_h).plus(zReg);
	}
	
	public void test(List<Datum> testData){
		System.out.println("===================");
		System.out.println("Test started...");
		int numCorrect = 0;
		
		System.out.println("test size = " + testData.size());
		
		int tp = 0, fp = 0;
		int fn = 0;
		int sentenceIndex = 0;
		
		for(int j=0; j<testData.size(); j++){
			
			LinkedList<Integer> wordIndexList = new LinkedList<Integer>();
			SimpleMatrix feature = new SimpleMatrix(windowSize * N_VECTOR_SIZE, 1);
			IntTuple tuple = extractTrainingExample(j, testData, feature, sentenceIndex, wordIndexList);
			int y = tuple.getFirst();
			sentenceIndex = tuple.getSecond();
			
			double hValue = getH(feature);
			
			int predict = hValue > 0.5 ? 1 : 0;
			if(predict == y) numCorrect++;
			
			if(predict == 1 && y == 1) tp++;
			else if(predict == 1 && y == 0) fp++;
			else if(predict == 0 && y == 1) fn++;				
		}
		
		double precision = (1.0 * tp) / (tp + fp);
		double recall = (1.0 * tp) / (tp + fn);
		double f1 = 2.0*(precision * recall) / (precision + recall);
		
		System.out.println("acc = " + (1.0*numCorrect/testData.size()));
		System.out.println("precision = " + precision);
		System.out.println("recall = " + recall);
		System.out.println("F1 = " + f1);
	}
}

class IntTuple{
	private int x;
	private int y;
	public IntTuple(int x, int y){
		this.x = x;
		this.y = y;
	}
	public int getFirst(){ return x;}
	public int getSecond(){ return y;}
}