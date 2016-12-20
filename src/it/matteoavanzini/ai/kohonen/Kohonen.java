package it.matteoavanzini.ai.kohonen;

public class Kohonen implements KohonenNetwork {

	@Override
	public double[][] perform(double[][] input) {
		// TODO Auto-generated method stub
		return null;
	}

	private double differenzaQuadrata(double a, double b) {
		return Math.pow(Math.abs(a - b), 2);
	}
	
	@Override
	public double calcolaDistanzaEuclidea(Node a, Node b) {
		double x = differenzaQuadrata(a.getX(), b.getX());
		double y = differenzaQuadrata(a.getY(), b.getY());
		double z = differenzaQuadrata(a.getZ(), b.getZ());
		double distanza = Math.sqrt(x + y + z);
		return distanza;
	}

	@Override
	public Node getBMU(double[][] input) {
		// TODO Auto-generated method stub
		return null;
	}

}
