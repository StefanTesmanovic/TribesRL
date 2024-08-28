import org.json.JSONObject;
import utils.file.IO;

public class Main {
    public static void main(String[] args){
        JSONObject conf = new IO().readJSON("training.json");
        if(conf.getBoolean("training")){
            Play.start();
        }else{
            System.out.println("Nije trening");
        }

    }
}
