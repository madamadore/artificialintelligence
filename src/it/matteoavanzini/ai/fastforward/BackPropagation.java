package it.matteoavanzini.ai.fastforward;

public abstract class BackPropagation extends FastForward implements NeuralBackPropagation {
	
	private final double ¢ = 0.001;
	protected double [][][] vecchiPesi;
	private double[][] derivate = new double[0][];

	public BackPropagation(Integer...strati) {
        super(strati);
        derivate = new double[strati.length][];
        vecchiPesi = pesi;
    }
	
	/**
     * @param ŋ         costante di apprendimento
     * @param errors    un array lungo quanto l'output che contiene per ciascun
     *                  output: (oX-tX), dove oX è l'output dell'unità X e tX è
     *                  il valore desiderato dall'unità X
     */
	@Override
    public void backPropagation(double ŋ, double µ, double [] errors) {

		int indiceUltimoStrato = strati.length-1;
		double [][] backPropagationError = new double[strati.length-1][];
        
		// STEP 1: calcolo del backPropagationError
        for (int indiceStratoCorrente=indiceUltimoStrato; indiceStratoCorrente>0; indiceStratoCorrente--) {
        		int lunghezzaStratoCorrente = strati[indiceStratoCorrente];
        		int lunghezzaStratoPrecedente = strati[indiceStratoCorrente-1];
        		
        		backPropagationError[indiceStratoCorrente-1] = new double[lunghezzaStratoCorrente];
        		
        		if (indiceStratoCorrente == indiceUltimoStrato) {
	        			for (int i=0; i<lunghezzaStratoCorrente; i++) {
	        				backPropagationError[indiceStratoCorrente-1][i] = (derivate[indiceUltimoStrato][i]+¢)*errors[i];
	        			}
        		} else {
        				for (int i=0; i<lunghezzaStratoCorrente; i++) {
        		            double somma = 0;
        		            for (int j=0; j<lunghezzaStratoPrecedente; j++) {
        		                somma += backPropagationError[indiceStratoCorrente-1][j]*pesi[indiceStratoCorrente][i][j];
        		            }
        		            backPropagationError[indiceStratoCorrente][i] = (derivate[indiceStratoCorrente][i]+¢)*somma;
        		        }
        		}
        
        }
        
        // STEP 2: aggiornamento dei pesi
        for (int indiceStratoCorrente = indiceUltimoStrato-1; indiceStratoCorrente>=0; indiceStratoCorrente--) {
	        
        		int lunghezzaStratoCorrente = strati[indiceStratoCorrente];
        		int lunghezzaStratoSuccessivo = strati[indiceStratoCorrente + 1];
        		
            for (int i=0; i<lunghezzaStratoCorrente; i++) {
                for (int j=0; j<lunghezzaStratoSuccessivo; j++) {
                    double delta = ŋ*sigma[lunghezzaStratoCorrente][i]*backPropagationError[indiceStratoCorrente][j];
                    pesi[indiceStratoCorrente][i][j] -= delta+(µ*vecchiPesi[indiceStratoCorrente][i][j]);
                    vecchiPesi[indiceStratoCorrente][i][j] = delta;
                }
            }
        }

    }
	
	@Override
	protected double[] feedForward(double [] inputs) {

        double [][] somme = new double[strati.length][0];
        
        somme[0] = inputs;
        for (int i=1; i<strati.length; i++) {
        		int dimension = strati[i] + 1;
        		if (i==strati.length-1) dimension = strati[i];
        		somme[i] = new double[dimension];
        		derivate[i] = new double[dimension];
        }

        for (int strato = 0; strato<strati.length-1; strato++) {
        		sigma[strato] = new double[pesi[strato].length];
	        	for (int i=0; i<pesi[strato].length; i++) {
	        		sigma[strato][i] = (i==pesi[strato].length-1) ? BIAS : sigma(somme[strato][i]);
	        		
	        		int dimension = strati[strato+1];
	        		for (int j=0; j<dimension; j++) {
	        			somme[strato+1][j] += ( sigma[strato][i]*pesi[strato][i][j] ) ; 
	        		}
	        	}
        }
        
        int indiceUltimoStrato = strati.length-1;
        for (int i=0; i<strati[indiceUltimoStrato]; i++) {
        		somme[indiceUltimoStrato][i] = sigma(somme[indiceUltimoStrato][i]);
        		derivate[indiceUltimoStrato][i] = derivata(somme[indiceUltimoStrato][i]);
        }
        
        return somme[strati.length-1];
    }
}
