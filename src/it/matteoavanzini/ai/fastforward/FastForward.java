package it.matteoavanzini.ai.fastforward;

import java.util.Random;
/**
 * @author emme
 */
public abstract class FastForward extends AbstractFastForward  {

    public FastForward(Integer...strati) {
        super(strati);
    }
    
        //               1
        //s3(x) = -----------------
        //          (1 + e^(−3x))
    protected double sigma(double input) {

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
    protected double derivata(double input) {
        double dividendo = Math.exp(input*C*-1);
        double divisore = Math.pow((1+dividendo), 2);
        double derivata = dividendo/divisore;
        if (derivata<=0.01) derivata = 0.01;
        return derivata;
    }
    
    protected abstract double normalize(double [] input);
    
}
