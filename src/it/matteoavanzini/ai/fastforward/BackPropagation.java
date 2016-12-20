package it.matteoavanzini.ai.fastforward;

public class BackPropagation extends FastForward implements NeuralBackPropagation {
	
	private final double ¢ = 0.001;
	/** derivata nascosto */
    private double[] dN = new double [LUNGHEZZA_NASCOSTO];
    /** derivata output */
    private double[] dO = new double [LUNGHEZZA_OUTPUT];

	/**
     * @param ŋ         costante di apprendimento
     * @param errors    un array lungo quanto l'output che contiene per ciascun
     *                  output: (oX-tX), dove oX è l'output dell'unità X e tX è
     *                  il valore desiderato dall'unità X
     */
	@Override
    public void backPropagation(double ŋ, double µ, double [] errors) {

		int lunghezzaPrimoStrato = strati[0];
		int lunghezzaUltimoStrato = strati[strati.length-1];
        double [] bpe1 = new double[LUNGHEZZA_NASCOSTO];
        double [] bpe2 = new double[lunghezzaUltimoStrato];
        // STEP 2: backpropagation to the output layer:
        for (int i=0; i<lunghezzaUltimoStrato; i++) {
            bpe2[i] = (dO[i]+¢)*errors[i];
        }
        // STEP 3: backpropagation to the hidden layer:
        for (int i=0; i<LUNGHEZZA_NASCOSTO; i++) {
            double somma = 0;
            for (int j=0; j<lunghezzaUltimoStrato; j++) {
                somma += bpe2[j]*w2[i][j];
            }
            bpe1[i] = (dN[i]+¢)*somma;
        }
        // STEP 4: weight updates:
        for (int i=0; i<LUNGHEZZA_NASCOSTO; i++) {
            for (int j=0; j<lunghezzaUltimoStrato; j++) {
                double delta = ŋ*sN[i]*bpe2[j];
                w2[i][j] -= delta+(µ*oldW2[i][j]);
                oldW2[i][j] = delta;
            }
        }
        for (int i=0; i<lunghezzaPrimoStrato; i++) {
            for (int j=0; j<LUNGHEZZA_NASCOSTO; j++) {
                double delta = ŋ*sI[i]*bpe1[j];
                w1[i][j] -= delta+(µ*oldW1[i][j]);
                oldW1[i][j] = delta;
            }
        }
    }
}
