package players;

import org.tensorflow.types.TFloat32;

public class Rewards {
    public TFloat32 probability;
    public float reward;
    public int index;
    public static float gamma = (float) 0.97;
    public float[] Gstate;


    public Rewards(int index, float reward, float[] state){
        this.index = index;
        this.reward = reward;
        Gstate = state;
    }

    public void setGamma(float gamma){ this.gamma = gamma;}
}
