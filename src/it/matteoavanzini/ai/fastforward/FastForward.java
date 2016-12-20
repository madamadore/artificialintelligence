package it.matteoavanzini.ai.fastforward;

import java.util.Random;
/**
 * @author emme
 */
public class FastForward implements NeuralNetwork {

	protected Integer[] strati;
	protected double [][][] pesi;
	
	public final int ID;

    private static int id = 0;
    private static double BIAS = 1;
    /** parametro c della funzione sigma */
    private static double C = 4;

    private double[] feedForward(double [] inputs) {

        double [][] somma = new double[strati.length][0];
        
        somma[0] = inputs;
        for (int i=1; i<strati.length; i++) {
        		int dimension = strati[i] + 1;
        		if (i==strati.length-1) dimension = strati[i];
        		somma[i] = new double[dimension];
        }

        for (int strato = 0; strato<strati.length-1; strato++) {
        		double sigma[] = new double[pesi[strato].length];
	        	for (int i=0; i<pesi[strato].length; i++) {
	        		sigma[i] = (i==pesi[strato].length-1) ? BIAS : sigma(somma[strato][i]);
	        		
	        		int dimension = strati[strato+1];
	        		for (int j=0; j<dimension; j++) {
	        			somma[strato+1][j] += ( sigma[i]*pesi[strato][i][j] ) ; 
	        		}
	        	}
        }
        
        for (int i=0; i<strati[strati.length-1]; i++) {
        		somma[strati.length-1][i] = sigma(somma[strati.length-1][i]);
        }
        return somma[strati.length-1];
    }
    
    private int getSerialDimension() {
		int dimension = 0;
	    	for (int i=0; i<strati.length; i++)  {
	    		for (int j=0; j<strati[i]; j++) {
	    			int dim = strati[i] + 1;
	        		if (i==strati.length-1) dim = strati[i];
	        		dimension += dim;
			}
		}
		return dimension;
	}
    
    /**
     * inizializza i pesi in modo casuale
     */
    private double[][][] initialize()
    {
        Random generator = new Random( System.currentTimeMillis() );
        int dimension = getSerialDimension();
        double [] seriale = new double[dimension];
        for (int i=0; i<seriale.length; i++) {
        		seriale[i] = (generator.nextDouble()*2.4)-1.2;
        }
        double [][][] ret = deserialize(seriale);
        return ret;
    }
    
    public double[][][] deserialize(double [] serialized) {
    		double [][][] retval = new double[strati.length-1][0][0];
        int index = 0;
        
        for (int i=0; i<retval.length; i++)  {
	        	retval[i] = new double[strati[i]+1][0];
	        	
	        	for (int j=0; j<retval[i].length; j++) {
	        		int dimension = strati[i+1] + 1;
	        		if (i==retval.length-1) dimension = strati[i+1];
	        		retval[i][j] = new double[dimension];
	        		
	        		for (int k=0; k<retval[i][j].length; k++) {
	        			retval[i][j][k] = serialized[index];
	        			index++;
	        		}
	    		}
        }
        return retval;
    }
    
    public double[] serialize(double [][][] pesi) {
    	
		int dimension = getSerialDimension();
		double ret[] = new double[dimension];
		
		int index = 0;
		for (int i=0; i<pesi.length; i++) {
			for (int j=0; j<pesi[i].length; j++) {
				for (int k=0; k<pesi[i][j].length; k++) {
					ret[index] = pesi[i][j][k];
					index++;
				}
			}
		}
		return ret;
	}

    public FastForward(Integer...strati) {
        this.ID = ++id;
        this.strati = strati;
        this.pesi = initialize();
    }
    
    @Override
    public Integer[] getStrati() { return strati; }
    
        //               1
        //s3(x) = -----------------
        //          (1 + e^(−3x))
    private double sigma(double input) {

        double exp = Math.exp(input*C*-1);
        double divisore = 1+exp;
        double sigma = (1 / divisore);
        
        return sigma;
    }

    // derivata della funzione sigma:
    //                     e^(−3x)
    //      s'3(x) = -------------------
    //                  (1 + e^(−3x))²
    /**
     * @param input il risultato della funzione sigma: s(x)
     */
    private double derivata(double input) {
        double dividendo = Math.exp(input*C*-1);
        double divisore = Math.pow((1+dividendo), 2);
        double derivata = dividendo/divisore;
        if (derivata<=0.01) derivata = 0.01;
        return derivata;
    }
    
    private double normalize(double [] input) {
    		String somma = "";
        for (int i=0; i<input.length; i++) {
        		somma += Integer.toString((int) input[i]);
        }
        return Integer.parseInt(somma, 2);
    }
    
	@Override
	public double[][][] getWeight() {
		return pesi;
	}
	
	public void setWeight(double [][][] pesi) {
		this.pesi = pesi;
	}
	
	@Override
	public double perform(double[] inputs) {
        double [] output = feedForward(inputs);
        double norm = normalize(output);
        return norm;
	}
}
