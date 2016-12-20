package it.matteoavanzini.ai.fastforward;

public interface NeuralNetwork {
	Integer[] getStrati();
	double [][][] getWeight();
	double perform(double [] input);
	double[] serialize(double[][][] pesi);
	double[][][] deserialize(double[] pesi);
}
