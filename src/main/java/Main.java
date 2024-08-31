import org.json.JSONObject;
import utils.file.IO;

public class Main {
    public static void main(String[] args){
        JSONObject conf = new IO().readJSON("training.json");
        if(conf.getBoolean("training")){
            for(int i = 0; i < 5000; i++)
                Play.start();
        }else{
            System.out.println("Nije trening");
        }

    }
}
