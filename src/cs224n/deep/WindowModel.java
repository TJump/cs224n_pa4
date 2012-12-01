package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {
	
	final double EPSILON = 1e-4;
	final double THRESHOLD = 1e-7;
	final boolean GRADCHECK = !true;
	final double REGULARIZATION_WEIGHT = 0.001;
	
	protected SimpleMatrix L, W, Wout, U, b1;
	protected double b2;
	double learningRate = 0.001;
	
	int numIterations = 3;
	
	public int windowSize, wordSize, hiddenSize;
	
	int n = 50;
	public WindowModel(int _windowSize, int _hiddenSize, double _lr){
		windowSize = _windowSize;
		hiddenSize = _hiddenSize;
		
		// Dimension(L) = n x V
		L = FeatureFactory.allVecs;
	}		

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		int fanIn = n * windowSize;
		int fanOut = hiddenSize;
		double epsilonInit = Math.sqrt(6.0 / (fanIn + fanOut));
		
		// Dimension(W) = H x C*n
		W = SimpleMatrix.random(hiddenSize, windowSize * n, -epsilonInit, epsilonInit, new Random());
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
		return getWordVector(startOrEnd ? "<s>" : "</s>");
	}
	
	
	private SimpleMatrix getWordVector(String s){
		if(FeatureFactory.wordToNum.containsKey(s)){
			int index = FeatureFactory.wordToNum.get(s);
			// false -- return column
			return L.extractVector(false, index);
		}
		else{
			// unknown word
			return L.extractVector(false, 0);
		}
	}
	
	int trainingSizeM = -1;
//	private double getCost()
	public void train(List<Datum> _trainData){
		
		System.out.println("training started...");
		long startTime = System.currentTimeMillis();
		
		trainingSizeM = _trainData.size();
		SimpleMatrix startVector = getStartOrEnd(true);
		SimpleMatrix endVector = getStartOrEnd(false);
		int halfWindowSize = windowSize/2;
		
		for(int i=0;i<numIterations;i++){
			System.out.println("training size = " + _trainData.size());
			
			double sum = 0;
			int sentenceIndex = 0;
			
			for(int j=0; j<_trainData.size(); j++){
				
				if(j%50000==0) System.out.println("#iter = " + j);
				
				//HACK
//				if(j>30000) break;
				
				Datum d = _trainData.get(j);
				String centerWord = d.word;
				String lable = d.label;
				int y = lable.equals("O") ? 0 : 1;
				
//				System.out.println(d.word + ", " + d.label);
				
				SimpleMatrix feature = new SimpleMatrix(windowSize * n, 1);
				Set<Integer> indexSet = new HashSet<Integer>();
				int numStartTagTofill = Math.max(0, halfWindowSize - sentenceIndex);
				for(int k=0; k<numStartTagTofill; k++){
//					System.out.print("<s> ");
					indexSet.add(FeatureFactory.wordToNum.get("<s>"));
					feature.insertIntoThis(k * n, 0, startVector);
				}
				
				int numVectorTofill = halfWindowSize - numStartTagTofill;
				for(int k=0;k<numVectorTofill;k++){
					Datum d_ = _trainData.get(j-numVectorTofill+k);
					String word_ = d_.word;
					indexSet.add(FeatureFactory.getIndex(word_));
//					System.out.print(word_ + " ");
					SimpleMatrix theVector = getWordVector(word_);
					feature.insertIntoThis((k + numStartTagTofill) * n, 0, theVector);
				}
				
				// over x... end of window size
				for(int k=0;k<=halfWindowSize;k++){
					
					Datum d_ = _trainData.get(j+k);
					String word_ = d_.word;
					indexSet.add(FeatureFactory.getIndex(word_));
//					System.out.print(word_ + " ");
					SimpleMatrix theVector = getWordVector(word_);
					feature.insertIntoThis((k + halfWindowSize) * n, 0, theVector);
					
					if(j+k == _trainData.size()-1 || word_.equals(".")){
						while(k<halfWindowSize){
							indexSet.add(FeatureFactory.wordToNum.get("</s>"));
//							System.out.print("</s>");
							feature.insertIntoThis((k + halfWindowSize) * n, 0, endVector);
							k++;
						}
						break;
					}
				}
				
				
//				System.out.println("\nh(x) = " + hValue);
				
				if(centerWord.equals(".")){
					sentenceIndex = 0;
				} else {
					sentenceIndex++;
				}
				
				sum += getCostJ(feature, y);
				
//				System.out.println("D_J_U = " + getD_J_U(feature, y));
//				System.out.println("D_J_W = " + getD_J_W(feature, y));
//				System.out.println();
//				System.out.println("sum error = " + sum);
				
				//-------------------------------------------------------
				// Gradient Check
				if(GRADCHECK){
					boolean check = gradientCheck(feature, y); 
					System.out.println("gradient check " + (check? "passed!" : "failed!") + "\n");
				}
				
				// SGD
				U = U.minus(getD_J_U(feature, y).scale(learningRate));
				W = W.minus(getD_J_W(feature, y).scale(learningRate));
				b1 = b1.minus(getD_J_b1(feature, y).scale(learningRate));
				b2 -=  getD_J_b2(feature, y) * learningRate;
				
				// update L
//				for(int indexL : indexSet){
//					SimpleMatrix vector = L.extractVector(false, indexL);
//					SimpleMatrix d_j_l = getD_J_L(vector, y).scale(learningRate);
//					for(int indexN=0;indexN<n;indexN++)
//						L.set(indexN, indexL, d_j_l.get(indexN, 0));
//				}
				
			}
			
			double totalCost_J = sum/_trainData.size();
			System.out.println("total cost = " + totalCost_J);
		}
		long endTime = System.currentTimeMillis();
		System.out.println("Training took " + (endTime - startTime)/1000 + " sec.");
	}
	
	enum PARAMETERS {U, W, B1, B2, L};
	private SimpleMatrix getApproximateGradient(SimpleMatrix feature, int y, PARAMETERS params){
		SimpleMatrix result = null;
		
		if(params == PARAMETERS.W){
			// W: H x c*n
			result = new SimpleMatrix(W);
			for(int i=0;i<hiddenSize;i++){
				for(int j=0;j<windowSize*n;j++){
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
			result = null;
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
			
			if(param == PARAMETERS.W) grad = getD_J_W(feature, y);
			else if(param == PARAMETERS.U) grad = getD_J_U(feature, y);
			else if(param == PARAMETERS.B1) grad = getD_J_b1(feature, y);
			else if(param == PARAMETERS.B2) {
				grad = new SimpleMatrix(1, 1);
				grad.set(0, 0, getD_J_b2(feature, y));
			}
			double delta = grad.minus(gradApprox).normF();
			System.out.println("\t" + param.toString() + " delta = " + delta);
			if(delta > THRESHOLD) return false;
		}
		return true;
	}
	private double getCostJ(SimpleMatrix x, int y){
		double hValue = getH(x);
		double regTerm = 0.5 * REGULARIZATION_WEIGHT * (W.elementMult(W).elementSum() + U.elementMult(U).elementSum());
		
		return -y * Math.log(hValue) + (y-1)*Math.log(1-hValue) + regTerm;
	}
	
	// dw(i,j) = sigmoid * (1-sigmoid) * [u(i,0) * D[tanh^2 (w(i,j) * x(j,0))] * x(j,0)]
	private SimpleMatrix getD_J_W(SimpleMatrix x, double y){
		SimpleMatrix dw = new SimpleMatrix(hiddenSize, windowSize * n);
		SimpleMatrix dwReg = new SimpleMatrix(hiddenSize, windowSize * n);
		
		SimpleMatrix z = getTanhDerivativeMatrix(W.mult(x).plus(b1));
		
		dw = z.elementMult(U).mult(x.transpose());
		
		for(int i=0;i<hiddenSize;i++)
			for(int j=0;j<windowSize * n;j++){
				dwReg.set(i, j, REGULARIZATION_WEIGHT * W.get(i,j)); 
			}
		
		double h = getH(x);				
		double factor = -y/h + (1-y)/(1-h);
		h *= 1-h;
		return dw.scale(factor * h).plus(dwReg);
	}
	
	private SimpleMatrix getD_J_L(SimpleMatrix x, double y){
		SimpleMatrix dL = new SimpleMatrix(windowSize * n, 1);
		
		for(int i=0;i<windowSize * n;i++){
			double sum = 0;
			for(int j=0;j<hiddenSize;j++)
				sum += U.get(j,0) * getTanhDerivative(W.get(j,i) * x.get(i,0)) * W.get(j, i);
			
			dL.set(i, 0, sum);
		}
		
		double h = getH(x);		
		double factor = y/h + (y-1)/(1-h);
		h *= 1-h;
		return dL.scale(-factor * h);
	}
	
	private double getD_J_b2(SimpleMatrix x, double y){
		double h = getH(x);		
		double factor = -y/h + (1-y)/(1-h);
		h *= 1-h;
		return factor * h;
	}
	
	// sigmoid * (1-sigmoid) * u(i,0) * (1-tanh^2 (b(i,0)))
	private SimpleMatrix getD_J_b1(SimpleMatrix x, double y){
		SimpleMatrix db = new SimpleMatrix(hiddenSize, 1);
		
		db = U.elementMult(getTanhDerivativeMatrix(W.mult(x).plus(b1)));
		
		double h = getH(x);
		double factor = -y/h + (1-y)/(1-h);
		h *= 1-h;
		return db.scale(factor * h);
	}
	
	// return sigmoid * (1-sigmoid) * tanh(wx+b1)
	private SimpleMatrix getD_J_U(SimpleMatrix x, double y){
		
		SimpleMatrix z = W.mult(x).plus(b1);
		SimpleMatrix zReg = U.scale(REGULARIZATION_WEIGHT);
		z = getTanhMatrix(z);
		double h = getH(x);
		double factor = y/h + (y-1)/(1-h);
		h *= 1-h;
		return z.scale(-factor * h).plus(zReg);
	}
	
	public void test(List<Datum> testData){
		SimpleMatrix startVector = getStartOrEnd(true);
		SimpleMatrix endVector = getStartOrEnd(false);
		int numCorrect = 0;
		int halfWindowSize = windowSize/2;
		
			System.out.println("test size = " + testData.size());
			
			int tp = 0, fp = 0;
			int fn = 0;
			int sentenceIndex = 0;
			
			for(int j=0; j<testData.size(); j++){
				
				Datum d = testData.get(j);
				String centerWord = d.word;
				String lable = d.label;
				int y = lable.equals("O") ? 0 : 1;
				
//				System.out.println(d.word + ", " + d.label);
				
				SimpleMatrix feature = new SimpleMatrix(windowSize * n, 1);
				Set<Integer> indexSet = new HashSet<Integer>();
				int numStartTagTofill = Math.max(0, halfWindowSize - sentenceIndex);
				for(int k=0; k<numStartTagTofill; k++){
//					System.out.print("<s> ");
					indexSet.add(FeatureFactory.wordToNum.get("<s>"));
					feature.insertIntoThis(k * n, 0, startVector);
				}
				
				int numVectorTofill = halfWindowSize - numStartTagTofill;
				for(int k=0;k<numVectorTofill;k++){
					Datum d_ = testData.get(j-numVectorTofill+k);
					String word_ = d_.word;
					indexSet.add(FeatureFactory.getIndex(word_));
//					System.out.print(word_ + " ");
					SimpleMatrix theVector = getWordVector(word_);
					feature.insertIntoThis((k + numStartTagTofill) * n, 0, theVector);
				}
				
				// over x... end of window size
				for(int k=0;k<=halfWindowSize;k++){
					
					Datum d_ = testData.get(j+k);
					String word_ = d_.word;
					indexSet.add(FeatureFactory.getIndex(word_));
//					System.out.print(word_ + " ");
					SimpleMatrix theVector = getWordVector(word_);
					feature.insertIntoThis((k + halfWindowSize) * n, 0, theVector);
					
					if(j+k == testData.size()-1 || word_.equals(".")){
						while(k<halfWindowSize){
							indexSet.add(FeatureFactory.wordToNum.get("</s>"));
//							System.out.print("</s>");
							feature.insertIntoThis((k + halfWindowSize) * n, 0, endVector);
							k++;
						}
						break;
					}
				}
				
				double hValue = getH(feature);
				
				int predict = hValue > 0.5 ? 1 : 0;
				if(predict == y) numCorrect++;
				
				if(predict == 1 && y == 1) tp++;
				else if(predict == 1 && y == 0) fp++;
				else if(predict == 0 && y == 1) fn++;
//				System.out.println("\nh(x) = " + hValue);
				
				if(centerWord.equals(".")){
					sentenceIndex = 0;
				} else {
					sentenceIndex++;
				}
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
