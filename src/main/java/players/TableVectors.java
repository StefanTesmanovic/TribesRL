package players;

import core.TribesConfig;
import utils.Vector2d;


//Mozda je brze da prvo izracunam za dolazenje iz uglova i kasnije da samo oduzimam ali onda se puno podataka cuva
public class TableVectors {
    public static float mSize = TribesConfig.DEFAULT_MAP_SIZE[1];
    public static float OneOverSize = 1/mSize;

    public static float[] Calculate(Vector2d source, Vector2d target){
        float[] ret = new float[8];
        if(source.x == target.x && source.y == target.y)
            for(int i = 0; i < 8; i++)
                ret[i] = 1;
        if(source.x >= target.x){
            if(source.y >= target.y){
                if(source.x - target.x >= source.y - target.y) {
                    ret[7] = source.y - target.y;
                    ret[0] = source.x - target.x - ret[7];
                    ret[7] = (mSize-ret[7]) * OneOverSize;
                    ret[0]  = (mSize-ret[0]) * OneOverSize;
                }else{
                    ret[7] = source.x - target.x;
                    ret[6] = source.y - target.y - ret[7];
                    ret[7] = (mSize-ret[7]) * OneOverSize;
                    ret[6] = (mSize-ret[6]) * OneOverSize;
                }
            }else{
                if(source.x - target.x >= target.y - source.y) {
                    ret[1] = target.y - source.y;
                    ret[0] = source.x - target.x - ret[1];
                    ret[1] = (mSize-ret[1]) * OneOverSize;
                    ret[0] = (mSize-ret[0]) * OneOverSize;
                }else{
                    ret[1] = source.x - target.x;
                    ret[2] = target.y - source.y - ret[1];
                    ret[1] = (mSize-ret[1]) * OneOverSize;
                    ret[2] = (mSize-ret[2]) * OneOverSize;
                }
            }
        }else{
            if(source.y >= target.y){
                if(target.x - source.x >= source.y - target.y) {
                    ret[5] = source.y - target.y;
                    ret[4] = target.x - source.x - ret[5];
                    ret[5] = (mSize-ret[5]) * OneOverSize;
                    ret[4] = (mSize-ret[4]) * OneOverSize;
                }else{
                    ret[5] = target.x - source.x;
                    ret[6] = source.y - target.y - ret[5];
                    ret[5] = (mSize-ret[5]) * OneOverSize;
                    ret[6] = (mSize-ret[6]) * OneOverSize;
                }
            }else{
                if(target.x - source.x >= target.y - source.y) {
                    ret[3] = target.y - source.y;
                    ret[4] = target.x - source.x - ret[3];
                    ret[3] = (mSize-ret[3]) * OneOverSize;
                    ret[4] = (mSize-ret[4]) * OneOverSize;
                }else{
                    ret[3] = target.x - source.x;
                    ret[2] = target.y - source.y - ret[3];
                    ret[3] = (mSize-ret[3]) * OneOverSize;
                    ret[2] = (mSize-ret[2]) * OneOverSize;
                }
            }
        }
        return ret;
    }
}
