import org.json.JSONObject;
import org.tensorflow.Session;
import players.RLAgent;
import utils.file.IO;

public class Main {


    public static void main(String[] args){
        RLAgent.initNN();
        JSONObject conf = new IO().readJSON("training.json");
        if(conf.getBoolean("training")){
            for(int i = 0; i < 1; i++)
                Play.start();
        }else{
            System.out.println("Nije trening");
        }

    }
}
