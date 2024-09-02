import org.json.JSONObject;
import org.tensorflow.Session;
import org.tensorflow.types.TFloat32;
import players.RLAgent;
import players.RLAgentTrain;
import players.Rewards;
import utils.file.IO;

import java.util.ArrayList;
import java.util.HashMap;

public class Main {


    public static void main(String[] args){
        RLAgent.initNN();
        RLAgentTrain.initNN();
        JSONObject conf = new IO().readJSON("training.json");
        if(conf.getBoolean("training")){
            for(int i = 0; i < 50; i++) {
                Play.start();
                for(int k : RLAgent.rewards.keySet()) {
                    ArrayList<Rewards> list = RLAgent.rewards.get(k);
                    float[] G = new float[list.size()];
                    G[list.size()-1] = list.get(list.size()-1).reward;
                    for (int j = list.size()-2; j >= 0; j--)
                        G[j] = Rewards.gamma * G[j+1] + list.get(j).reward;
                    for(int z = 0; z < list.size(); z++){
                        float[] ac = new float[RLAgent.ActionSpaceSize];
                        ac[list.get(z).index] = 1;
                        RLAgent.session.runner()
                                .feed(RLAgent.stateInput.asOutput(), list.get(z).Gstate)
                                .feed(RLAgent.actions.asOutput(), TFloat32.vectorOf(ac))
                                .feed(RLAgent.rew.asOutput(), TFloat32.scalarOf(G[z])) // Feed rewards during the backward pass
                                .addTarget(RLAgent.minimize)
                                .run();
                    }

                }
                RLAgent.rewards = new HashMap<>();
            }
        }else{
            System.out.println("Nije trening");
        }

    }
}
