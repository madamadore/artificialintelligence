package it.matteoavanzini.ai.fastforward;

import java.util.Random;
/**
 * @author emme
 */
public abstract class AbstractFastForward implements NeuralNetwork {

	protected Integer[] strati;
	protected double [][][] pesi;
	protected double sigma[][];
	
	public final int ID;

	protected static int id = 0;
    protected static double BIAS = 1;
    /** parametro c della funzione sigma */
    protected static double C = 4;

    protected double[] feedForward(double [] inputs) {

        double [][] somma = new double[strati.length][0];
        
        somma[0] = inputs;
        for (int i=1; i<strati.length; i++) {
        		int dimension = strati[i] + 1;
        		if (i==strati.length-1) dimension = strati[i];
        		somma[i] = new double[dimension];
        }

        for (int strato = 0; strato<strati.length-1; strato++) {
        		sigma[strato] = new double[pesi[strato].length];
	        	for (int i=0; i<pesi[strato].length; i++) {
	        		sigma[strato][i] = (i==pesi[strato].length-1) ? BIAS : sigma(somma[strato][i]);
	        		
	        		int dimension = strati[strato+1];
	        		for (int j=0; j<dimension; j++) {
	        			somma[strato+1][j] += ( sigma[strato][i]*pesi[strato][i][j] ) ; 
	        		}
	        	}
        }
        
        for (int i=0; i<strati[strati.length-1]; i++) {
        		somma[strati.length-1][i] = sigma(somma[strati.length-1][i]);
        }
        return somma[strati.length-1];
    }
    
    protected int getSerialDimension() {
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
    protected double[][][] initialize()
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

    public AbstractFastForward(Integer...strati) {
        this.ID = ++id;
        this.strati = strati;
        this.pesi = initialize();
        this.sigma = new double[strati.length][];
    }
    
    @Override
    public Integer[] getStrati() { return strati; }
    

    protected abstract double sigma(double input);

    protected abstract double derivata(double input);
    
    protected abstract double normalize(double [] input);
    
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
