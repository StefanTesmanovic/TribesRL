package players;

import core.Types;
import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.actions.unitactions.*;
import core.actions.unitactions.factory.UpgradeFactory;
import core.actors.City;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import org.tensorflow.GraphOperation;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.framework.op.sets.Sets;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;
import utils.ElapsedCpuTimer;
import utils.Vector2d;

import java.util.*;


public class RLAgentPlay extends Agent {

    private Random rnd;
    public static int ActionSpaceSize = 52, StateSpaceSize = 8*7;
    public static SavedModelBundle model;
    public RLAgentPlay(long seed)
    {
        super(seed);
        rnd = new Random(seed);
        String modelPath = "./modelTest-50";//"./modeli/model-500Turns-tanh-0.01-4250"; // Replace X with your version
        model = SavedModelBundle.load(modelPath, "serve");
        System.out.println(model.signatures());
        System.out.println(model.graph().operations());
        for (Iterator<GraphOperation> it = model.graph().operations(); it.hasNext(); ) {
            GraphOperation op = it.next();
            System.out.println(op.name() + " : " + op.type());
        }
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        SimpleAgentTrain temp = new SimpleAgentTrain(seed);
        //Gather all available actions:
        ArrayList<Action> allActions = gs.getAllAvailableActions();
        int nActions = allActions.size();
        //Initially pick a random action so that at least that can be returned
        //Action bestAction = allActions.get(m_rnd.nextInt(nActions));
        int bestActionScore = -1; //evalAction(gs,bestAction);

        HashMap<Integer, ArrayList<Action>> desiredActions = new HashMap<>();
        HashSet<Integer> units = new HashSet<Integer>();


        for (Action a : allActions) {
            if (a instanceof UnitAction)
                continue;
            int actionScore = temp.evalAction(gs, a);

            ArrayList<Action> listActions;
            if (!desiredActions.containsKey(actionScore)) {
                listActions = new ArrayList<>();
                desiredActions.put(actionScore, listActions);
            } else {
                listActions = desiredActions.get(actionScore);
            }
            listActions.add(a);

            if (actionScore > bestActionScore) {
//                bestAction = a;
                bestActionScore = actionScore;
            }
        }

        Action chosenAction = null;
        boolean actionFound = false;
        int val = bestActionScore;
        while (!actionFound && val >= -1) {
            if (desiredActions.containsKey(val)) {
                actionFound = true;
                int n = desiredActions.get(val).size();
                chosenAction = desiredActions.get(val).get(rnd.nextInt(n));
            } else {
                val--;
            }
        }
        if (chosenAction.getActionType() != Types.ACTION.END_TURN)
            return chosenAction;
        for (Action a : allActions) {
            if (!(a instanceof UnitAction)) continue;
            if (!units.contains(((UnitAction) a).getUnitId())) {
                units.add((((UnitAction) a).getUnitId()));
            }
        }
        Action action = new EndTurn();
        if(units == null) return action;
        for (int unID : units) {
            float[] input = InputNew(unID, gs);
            //int rand = m_rnd.nextInt(7 * 7 + 3);//plus 3 for upgrade, recover and disband
            TFloat32 mrtviTenzor = TFloat32.tensorOf(Shape.of(1, input.length), data -> {
                for (int i = 0; i < input.length; i++) {
                    data.setFloat((float) input[i], 0, i);
                }
            });
            /**mrtviTenzor = tf.reshape(
             tf.dtypes.cast(tf.constant(input), TFloat32.class),
             tf.array(-1L, input.length)
             ).asTensor();**/
            TFloat32 actionProbs = (TFloat32) model.session().runner()
                    .feed("input" ,mrtviTenzor)
                    .fetch("probabilities")
                    .run()
                    .get(0);
            //NdArray arr = NdArrays.ofFloats(Shape.of(1,51));
            Integer[] outputIndexes = sortedArgs(actionProbs);
            /**float a = 0;
             for(int i = 0; i < actionProbs.size(); i++) {
             System.out.print(actionProbs.getFloat(0, i));
             a += actionProbs.getFloat(0,i);
             }
             System.out.println("\n" + a);**/
            //System.out.println(actionProbs.size());//actionProbs.copyTo(arr));//.getFloat(0,2));

            int ind;
            for(int k = 0; k < ActionSpaceSize; k++){
                action = outputAction(unID, gs, outputIndexes[k]);
                if(action != null)
                    return action;
            }
            return new EndTurn();
        }
        return action;
    }
    private static float[] InputNew(int unitID, GameState gs){
        float[] ret = new float[StateSpaceSize], temp;
        int eDef=0, eAt = 8, fAt = 8*2, village = 8*3, ruins = 8*4, fCity = 8*5, eCity = 8*6;
        Unit actor = (Unit) gs.getActor(unitID);
        int tribeID = actor.getTribeId();
        Vector2d position = actor.getPosition(), target;
        Board board = gs.getBoard();
        float multiplier;
        for(int i = 0; i < TableVectors.mSize; i++){
            for(int j = 0; j < TableVectors.mSize; j++){
                target = new Vector2d(i, j);
                Types.TERRAIN teren = board.getTerrainAt(i,j);
                if(teren == Types.TERRAIN.VILLAGE){
                    temp = TableVectors.Calculate(position, target);
                    for(int k = 0; k < 8; k++) {
                        if(ret[k+village] > temp[k])
                            continue;
                        ret[k + village] = temp[k];
                    }
                }else if(board.getResourceAt(i,j) == Types.RESOURCE.RUINS){
                    temp = TableVectors.Calculate(position, target);
                    for(int k = 0; k < 8; k++) {
                        if(ret[k+ruins] > temp[k])
                            continue;
                        ret[k + ruins] = temp[k];
                    }
                }else if(teren == Types.TERRAIN.CITY){
                    int city = board.getCityIdAt(i,j);
                    City grad = (City) gs.getActor(city);
                    multiplier = 1;
                    if(grad.hasWalls()) multiplier = (float)0.7;
                    int trId = grad.getTribeId();
                    if(trId != tribeID) {
                        temp = TableVectors.Calculate(position, target);
                        for(int k = 0; k < 8; k++) {
                            if(ret[k+eCity] > temp[k]*multiplier)
                                continue;
                            ret[k + eCity] = temp[k]*multiplier;
                        }
                    }else{
                        temp = TableVectors.Calculate(position, target);
                        for(int k = 0; k < 8; k++) {
                            if(ret[k+fCity] > temp[k])
                                continue;
                            ret[k + fCity] = temp[k];
                        }
                    }
                }
                if(i == position.x && j == position.y)
                    continue;
                Unit unit = board.getUnitAt(i,j);
                if(unit == null) continue;
                if(unit.getTribeId() != tribeID){
                    multiplier = (float)actor.ATK/((float)actor.ATK*((float)actor.getCurrentHP()/(float)actor.getMaxHP()) + unit.DEF*((float)unit.getCurrentHP()/(float)unit.getMaxHP()));
                    temp = TableVectors.Calculate(position, target);
                    for(int k = 0; k < 8; k++) {
                        if(ret[k+eDef] > temp[k]*multiplier)
                            continue;
                        ret[k + eDef] = temp[k]*multiplier;
                    }
                    multiplier = (float)actor.DEF/((float)actor.DEF*((float)actor.getCurrentHP()/(float)actor.getMaxHP()) + (float)unit.ATK*((float)unit.getCurrentHP()/(float)unit.getMaxHP()));
                    temp = TableVectors.Calculate(position, target);
                    for(int k = 0; k < 8; k++) {
                        if(ret[k+eAt] > temp[k]*multiplier)
                            continue;
                        ret[k + eAt] = temp[k]*multiplier;
                    }
                }else{
                    multiplier = ((float)unit.ATK*(float)unit.getCurrentHP()/(float)unit.getMaxHP())/5;// 5 is max attack of all units (super unit attack)
                    temp = TableVectors.Calculate(position, target);
                    for(int k = 0; k < 8; k++) {
                        if(ret[k+fAt] > temp[k]*multiplier)
                            continue;
                        ret[k + fAt] = temp[k]*multiplier;
                    }
                }

            }
        }


        return ret;
    }
    private Action outputAction(int uId, GameState gs, int ouptutNeuron){
        if(ouptutNeuron == 49){
            LinkedList<Action> list = (new UpgradeFactory()).computeActionVariants(gs.getActor(uId), gs);
            if(!list.isEmpty())
                return list.get(0);
            return null;
        }else if(ouptutNeuron == 50){
            Action a = new Recover(uId);
            if(a.isFeasible(gs))
                return a;
            return null;
        }else if(ouptutNeuron == 51){
            Action a = new Disband(uId);
            if(a.isFeasible(gs))
                return a;
            return null;
        }else{
            Board board = gs.getBoard();
            Unit thisUnit = (Unit)gs.getActor(uId);
            int uX = (gs.getActor(uId)).getPosition().x, uY = (gs.getActor(uId)).getPosition().y;
            int x = ouptutNeuron/7+uX-3, y = ouptutNeuron%7+uX-3;
            if(x < 0 || y < 0 || x >= gs.getBoard().getSize() || y >= gs.getBoard().getSize()) return null;
            if(board.getResourceAt(x,y) == Types.RESOURCE.RUINS && (new Examine(uId)).isFeasible(gs)) return new Examine(uId);
            if(board.getTerrainAt(x,y) == Types.TERRAIN.VILLAGE){
                Capture a = new Capture(uId);
                a.setCaptureType(Types.TERRAIN.VILLAGE);
                if(a.isFeasible(gs)) return a;
            }
            if(board.getTerrainAt(x,y) == Types.TERRAIN.CITY){
                Capture a = new Capture(uId);
                a.setCaptureType(Types.TERRAIN.CITY);
                a.setTargetCity(board.getCityIdAt(x,y));
                if(a.isFeasible(gs)){System.out.println(a); return a;}
            }
            Unit chosenUnit = board.getUnitAt(x,y);
            if(chosenUnit != null)
                if(chosenUnit.getTribeId() != thisUnit.getTribeId())
                    if(thisUnit.getType() == Types.UNIT.MIND_BENDER) {
                        Convert a = new Convert(uId);
                        a.setTargetId(chosenUnit.getActorId());
                        if (a.isFeasible(gs))
                            return a;
                        return null;
                    }else{
                        Attack a = new Attack(uId);
                        a.setTargetId(chosenUnit.getActorId());
                        if (a.isFeasible(gs))
                            return a;
                        return null;
                    }
                else
                if(thisUnit.getType() == Types.UNIT.MIND_BENDER) {
                    HealOthers a = new HealOthers(uId);
                    if (a.isFeasible(gs))
                        return a;
                    return null;
                }else
                    return null;

            Move a = new Move(uId);
            a.setDestination(new Vector2d(x, y));
            if(a.isFeasible(gs))return a;
            return null;
        }
    }

    public static Integer[] sortedArgs(TFloat32 arr) {
        int size = (int)arr.size();  // Assuming 1D NdArray
        // Create an array of indices
        Integer[] indices = new Integer[size];
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }

        // Sort indices based on the values in the array, in descending order
        Arrays.sort(indices, new Comparator<Integer>() {
            @Override
            public int compare(Integer i1, Integer i2) {
                return Float.compare(arr.getFloat(0,i2), arr.getFloat(0, i1));
            }
        });

        return indices;
    }
    @Override
    public Agent copy() {
        return null;
    }
}
