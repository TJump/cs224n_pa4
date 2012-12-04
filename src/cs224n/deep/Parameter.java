package cs224n.deep;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import org.ejml.simple.SimpleMatrix;

public class Parameter implements Serializable{

	private static final long serialVersionUID = -6363284946870835511L;
	private SimpleMatrix L, W, U, b1;
	
	
	public Parameter(SimpleMatrix L, SimpleMatrix W, SimpleMatrix U, SimpleMatrix b1, double b2){
		this.L = L;
		this.W = W;
		this.U = U;
		this.b1 = b1;
		this.b2 = b2;
	}
	
	public static String DEFAULT_FILE_NAME = "param.dat";
	
	public Parameter(){
		this(DEFAULT_FILE_NAME);
	}
	
	public Parameter(String fileName){
		load(fileName);
	}
	
	private void load(String fileName){
		
		try {
			L = SimpleMatrix.loadBinary("L_" + fileName);
			W = SimpleMatrix.loadBinary("W_" + fileName);
			U = SimpleMatrix.loadBinary("U_" + fileName);
			b1 = SimpleMatrix.loadBinary("B1_" + fileName);
			SimpleMatrix b2_ = SimpleMatrix.loadBinary("B2_" + fileName);
			b2 = b2_.get(0,0);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		System.out.println("Parameters are loaded from *." + fileName);
	}
	
	public void save(){
		save(DEFAULT_FILE_NAME);
	}
	
	public void save(String fileName){
		
		try {
			L.saveToFileBinary("L_" + fileName);
			W.saveToFileBinary("W_" + fileName);
			U.saveToFileBinary("U_" + fileName);
			b1.saveToFileBinary("B1_" + fileName);
			SimpleMatrix b2_ = new SimpleMatrix(1,1);
			b2_.set(0,0, b2);
			b2_.saveToFileBinary("B2_" + fileName);
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("parameters saved...");
	}
	
	public SimpleMatrix getL() {
		return L;
	}

	public void setL(SimpleMatrix l) {
		L = l;
	}

	public SimpleMatrix getW() {
		return W;
	}

	public void setW(SimpleMatrix w) {
		W = w;
	}

	public SimpleMatrix getU() {
		return U;
	}

	public void setU(SimpleMatrix u) {
		U = u;
	}

	public SimpleMatrix getB1() {
		return b1;
	}

	public void setB1(SimpleMatrix b1) {
		this.b1 = b1;
	}

	public double getB2() {
		return b2;
	}

	public void setB2(double b2) {
		this.b2 = b2;
	}

	private double b2;
	

}