package cs224n.deep;

public class IntTuple{
	private int x;
	private int y;
	public IntTuple(int x, int y){
		this.x = x;
		this.y = y;
	}
	public int getFirst(){ return x;}
	public int getSecond(){ return y;}
}