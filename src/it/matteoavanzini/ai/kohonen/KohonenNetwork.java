package it.matteoavanzini.ai.kohonen;

public interface KohonenNetwork {
	double calcolaDistanzaEuclidea(Node a, Node b);
	Node getBMU(double [][] input);
	double[][] perform(double[][] input);
}
