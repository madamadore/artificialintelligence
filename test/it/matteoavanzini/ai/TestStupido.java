package it.matteoavanzini.ai;

import it.matteoavanzini.ai.fastforward.BackPropagation;

public class TestStupido {

	public static void main(String [] args) {
		BackPropagation ff = new BackPropagation(8,4,2) {
			protected double normalize(double [] output) {
				double somma = 0;
				for (int i=0; i<output.length; i++) {
					somma += output[i];
				}
				return somma;
			}
		};
		for (int i = 0; i<100; i++) {
			double [] input = getInputCasuale(8);
			double x = ff.perform(input);
			System.out.println(x);
		}
	}
	
	private static double[] getInputCasuale(int lunghezzaInput) {
		double [] valoreDiRitorno = new double[lunghezzaInput];
		for (int i=0; i<valoreDiRitorno.length; i++) {
			double rnd = Math.random();
			valoreDiRitorno[i] = rnd;
		}
		return valoreDiRitorno;
	}
}
