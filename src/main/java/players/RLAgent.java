package players;


import com.google.protobuf.InvalidProtocolBufferException;
import core.Types;
import core.actions.Action;
import core.actions.tribeactions.EndTurn;
import core.actions.unitactions.*;
import core.actions.unitactions.command.AttackCommand;
import core.actions.unitactions.factory.UpgradeFactory;
import core.actors.Actor;
import core.actors.City;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import org.json.JSONObject;
import org.tensorflow.*;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.proto.ConfigProto;
import org.tensorflow.proto.GPUOptions;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TString;
import utils.ElapsedCpuTimer;
import utils.Pair;
import utils.Vector2d;
import utils.file.IO;


import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import static org.tensorflow.op.core.ReduceSum.keepDims;
public class RLAgent extends Agent{
    //public static Placeholder<TFloat32> movesCount;
    public static Graph graph = null;
    public static Ops tf;

    public static Placeholder<TFloat32> stateInput;
    public static Operand<TFloat32> logits, probabilities;
    public static Operand<TFloat32> actionProbabilities;

    public static Variable<TFloat32> weights1;
    public static Variable<TFloat32> biases1;
    public static Operand<TFloat32> layer1;

    public static Variable<TFloat32> weights2;
    public static Variable<TFloat32> biases2;

    public static Variable<TFloat32> weights3;
    public static Variable<TFloat32> biases3;

    public static Variable<TFloat32> weights4;
    public static Variable<TFloat32> biases4;

    public static int ActionSpaceSize = 52, StateSpaceSize = 8*7;
    public static Session session = null;
    public static Operand<TFloat32> copying;
    private Random m_rnd;
    public static Gradients gradients;
    public static Optimizer optimizer;
    public static Placeholder<TFloat32> actions;
    public static Placeholder<TFloat32> rew;
    public static Op minimize;
    public static boolean training = false;
    public static HashMap<Integer, ArrayList<Rewards>> rewards = new HashMap<>();

    public RLAgent(long seed)
    {
        super(seed);
        m_rnd = new Random(seed);
        JSONObject conf = new IO().readJSON("training.json");
        if(Objects.equals(conf.getString("runMode"), "Training")) training = true;
        if(Objects.equals(conf.getString("runMode"), "Testing")){
            if(session != null)
                session.close();
            if(graph != null)
                graph.close();
            initNN();
            try {
                loadModel("./modeli/model-tanh-500turns-gamma98-01-4500");
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
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
                chosenAction = desiredActions.get(val).get(m_rnd.nextInt(n));
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
            TFloat32 actionProbs = (TFloat32) session.runner()
                    .feed(stateInput.asOutput(), mrtviTenzor)
                    .fetch(actionProbabilities)
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
            int stateRew = rewardFromState(unID, gs);
            double prob;
            int ind;
            for(int k = 0; k < ActionSpaceSize; k++){
                prob = m_rnd.nextDouble();
                if(training && prob > 0.75){
                    ind = m_rnd.nextInt(k, ActionSpaceSize);
                    action = outputAction(unID, gs, outputIndexes[ind]);
                    if(action != null){
                        if(!rewards.containsKey(unID))
                            rewards.put(unID, new ArrayList<Rewards>());
                        ArrayList<Rewards> tmp = rewards.get(unID);
                        tmp.add(new Rewards(outputIndexes[ind], (new SimpleAgentTrain(seed)).evalAction(gs, action) + stateRew, input));
                        return action;
                    }else{
                        outputIndexes[ind] = outputIndexes[k];
                    }
                }
                action = outputAction(unID, gs, outputIndexes[k]);
                if(action != null){
                    if(!rewards.containsKey(unID))
                        rewards.put(unID, new ArrayList<Rewards>());
                    ArrayList<Rewards> tmp = rewards.get(unID);
                    tmp.add(new Rewards(outputIndexes[k], (new SimpleAgentTrain(seed)).evalAction(gs, action) + stateRew, input));
                    return action;
                }//{ System.out.println(gs.getTick() + ":" +gs.getActiveTribeID()+ ":" +action); return action;}
            }
            return new EndTurn();
        }
        return action;
    }
    //friendly Hp, friendly attack, enemy HP, enemy attack enemy--> HP is defence succes if attacked attack is how succesful would our attack be
    private static float[] Input(int unitId, GameState gs){
        int Range = 3, x, y;//7 = 3 + 1 + 3
        float[] ret = new float[7*7*8];NdArrays.ofFloats(Shape.of(7 * 7 * 8));
        Unit actor = (Unit) gs.getActor(unitId);
        int tribeID = actor.getTribeId();
        Vector2d position = actor.getPosition();
        Board board = gs.getBoard();
        Actor temp;
        int friendlyAttack = 7*7, enemyHP = 7*7*2, enemyAttack = 7*7*3, friendlyCity = 7*7*4, enemyCity = 7*7*5, res = 7*7*6, terr = 7*7*7;

        for(int i = 0; i < 7*7; i++){
            x = ((int) i/7)+ position.x - Range;
            y = i%7 + position.y - Range;
            if((x  >= 0 && y  >= 0) && (x < board.getSize() && y < board.getSize())) {
                temp = board.getUnitAt(x, y);
                if (temp != null) {
                    if (x == position.x && y == position.y) continue;
                    if (temp.getTribeId() != tribeID) {
                        Attack attack = new Attack(unitId);
                        attack.setTargetId(temp.getActorId());
                        AttackCommand ac = new AttackCommand();
                        Pair<Integer, Integer> pair = ac.getAttackResults(attack, gs);
                        ret[i+enemyHP] = pair.getSecond();//.setFloat((float)pair.getSecond(), i+enemyHP);
                        ret[i+enemyAttack] = pair.getFirst();//.setFloat((float)pair.getFirst(), i+enemyAttack);
                    } else {
                        temp = board.getUnitAt(x, y);
                        ret[i] = ((Unit)temp).getCurrentHP();//.setFloat((float)((Unit)temp).getCurrentHP(), i);
                        ret[i+friendlyAttack] = ((Unit)temp).ATK;// .setFloat((float)((Unit)temp).ATK, i + friendlyAttack);

                    }
                }
                temp = gs.getActor(board.getCityIdAt(x, y));

                if(temp != null)
                    if(temp.getTribeId() != tribeID)
                        ret[i+enemyCity] = ((City)temp).getBound();//.setFloat((float)((City)temp).getBound(), i + enemyCity);
                    else
                        ret[i + friendlyCity] = ((City)temp).getBound();//.setFloat((float)((City)temp).getBound(), i + friendlyCity);

                if(board.getResourceAt(x,y) == Types.RESOURCE.RUINS){ret[i+res] = 1;}//.setFloat((float)1, i + res);}
                if(board.getTerrainAt(x,y) == Types.TERRAIN.VILLAGE){ret[i+terr] = 1;}//.setFloat((float)1, i + terr);}

            }
        }
        return ret;
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

    private static int rewardFromState(int unitID, GameState gs) {
        int ret = 0;
        Unit actor = (Unit) gs.getActor(unitID);
        int tribeID = actor.getTribeId();
        Board board = gs.getBoard();
        ArrayList<Integer> cities = gs.getTribe(tribeID).getCitiesID();
        for(Integer c : cities) {
            City city = (City) gs.getActor(c);
            Vector2d position =  city.getPosition();
            for (int i = 0; i < 11*11; i++) {
                int x = ((int) i/7)+ position.x - 5;
                int y = i%7 + position.y - 5;
                if(!((x  >= 0 && y  >= 0) && (x < board.getSize() && y < board.getSize()))) continue;
                int dist = (int) Vector2d.chebychevDistance(position, new Vector2d(x, y));
                Unit u = board.getUnitAt(x, y);
                if(u == null) continue;
                if(u.getTribeId() != tribeID){
                    ret += dist <= 5 ? (dist <= 4 ? (dist <= 3 ? (dist <= 2 ? (dist <= 1 ? -100 : -50) : -10) : -5) : -1) : 0; // ( : <3 ternary operators <3 : )
                }
            }
        }
        return ret;
    }

    private double[] normaliseInput(double[] Input){
        return new double[8];
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
            int x = ouptutNeuron/7+uX-3, y = ouptutNeuron%7+uY-3;
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

    public static void initNN(){
        ConfigProto.Builder configBuilder = ConfigProto.newBuilder();
        GPUOptions.Builder gpuOptionsBuilder = GPUOptions.newBuilder();

        // Set GPU memory fraction, allow growth, etc. (optional configurations)
        gpuOptionsBuilder.setAllowGrowth(true);
        configBuilder.setGpuOptions(gpuOptionsBuilder);

        int numActions = ActionSpaceSize, input = StateSpaceSize;
        graph = new Graph();
        tf = Ops.create(graph);
        // Define the input placeholder (state input)
        // Variable to accumulate loss
        //movesCount = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1)));

        stateInput = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, input)));
        actions = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, numActions)));
        rew = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1)));

        weights1 = tf.variable(tf.random.truncatedNormal(tf.constant(new long[]{input, 200}), TFloat32.class));
        biases1 = tf.variable(tf.zeros(tf.constant(new long[]{200}), TFloat32.class));
        layer1 = tf.math.tanh(tf.math.add(tf.linalg.matMul(stateInput, weights1), biases1));


        weights2 = tf.variable(tf.random.truncatedNormal(tf.constant(new long[]{200, 150}), TFloat32.class));
        biases2 = tf.variable(tf.zeros(tf.constant(new long[]{150}), TFloat32.class));
        Operand<TFloat32> layer2 = tf.math.tanh(tf.math.add(tf.linalg.matMul(layer1, weights2), biases2));


        weights3 = tf.variable(tf.random.truncatedNormal(tf.constant(new long[]{150, 100}), TFloat32.class));
        biases3 = tf.variable(tf.zeros(tf.constant(new long[]{100}), TFloat32.class));
        Operand<TFloat32> layer3 = tf.math.tanh(tf.math.add(tf.linalg.matMul(layer2, weights3), biases3));


        weights4 = tf.variable(tf.random.truncatedNormal(tf.constant(new long[]{100, numActions}), TFloat32.class));
        biases4 = tf.variable(tf.zeros(tf.constant(new long[]{numActions}), TFloat32.class));

        logits = tf.math.add(tf.linalg.matMul(layer3, weights4), biases4);
        //logits = tf.math.add(logits, tf.reduceMin(logits, tf.constant(1)));

        actionProbabilities = tf.math.div(logits,tf.reshape(tf.reduceSum(logits, tf.array(1)), tf.array(-1, 1)));//tf.nn.softmax(logits);

        Operand<TFloat32> logProbs = tf.math.log(tf.reduceSum(tf.math.mul(actionProbabilities, actions), tf.constant(1)));
        Operand<TFloat32> loss = tf.math.neg(tf.math.mul(logProbs, rew));

        Optimizer optimizer = new Adam(graph, 0.01f);//Adam.createAdamMinimize(tf, 0.001f)//.create(tf, 0.001f);
        minimize = optimizer.minimize(loss);

        session = new Session(graph, configBuilder.build());
    }

    @Override
    public Agent copy() {
        return null;
    }
    public static void loadModel(String filePath) throws IOException {

        float[][] weights1 = new float[8*7][200];
        float[] biases1 = new float[200];
        float[][] weights2 = new float[200][150];
        float[] biases2 = new float[150];
        float[][] weights3 = new float[150][100];
        float[] biases3 = new float[100];
        float[][] weights4 = new float[100][52];
        float[] biases4 = new float[52];

        BufferedReader reader = new BufferedReader(new FileReader(filePath));
        String line;

        // Skip "weights1:" line
        line = reader.readLine();
        // Load weights1
        for (int i = 0; i < 8*7; i++) {
            line = reader.readLine();
            String[] values = line.split(" ");
            for (int j = 0; j < 200; j++) {
                weights1[i][j] = Float.parseFloat(values[j]);
            }
        }

        // Skip "biases1:" line
        line = reader.readLine();
        // Load biases1
        line = reader.readLine();
        String[] values = line.split(" ");
        for (int i = 0; i < 200; i++) {
            biases1[i] = Float.parseFloat(values[i]);
        }

        // Skip "weights2:" line
        line = reader.readLine();
        // Load weights2
        for (int i = 0; i < 200; i++) {
            line = reader.readLine();
            values = line.split(" ");
            for (int j = 0; j < 150; j++) {
                weights2[i][j] = Float.parseFloat(values[j]);
            }
        }

        // Skip "biases2:" line
        line = reader.readLine();
        // Load biases2
        line = reader.readLine();
        values = line.split(" ");
        for (int i = 0; i < 150; i++) {
            biases2[i] = Float.parseFloat(values[i]);
        }

        // Skip "weights3:" line
        line = reader.readLine();
        // Load weights3
        for (int i = 0; i < 150; i++) {
            line = reader.readLine();
            values = line.split(" ");
            for (int j = 0; j < 100; j++) {
                weights3[i][j] = Float.parseFloat(values[j]);
            }
        }

        // Skip "biases3:" line
        line = reader.readLine();
        // Load biases3
        line = reader.readLine();
        values = line.split(" ");
        for (int i = 0; i < 100; i++) {
            biases3[i] = Float.parseFloat(values[i]);
        }

        // Skip "weights4:" line
        line = reader.readLine();
        // Load weights4
        for (int i = 0; i < 100; i++) {
            line = reader.readLine();
            values = line.split(" ");
            for (int j = 0; j < 52; j++) {
                weights4[i][j] = Float.parseFloat(values[j]);
            }
        }

        // Skip "biases4:" line
        line = reader.readLine();
        // Load biases4
        line = reader.readLine();
        values = line.split(" ");
        for (int i = 0; i < 52; i++) {
            biases4[i] = Float.parseFloat(values[i]);
        }

        reader.close();
        TFloat32 mrtviTenzor;
        mrtviTenzor = TFloat32.tensorOf(Shape.of(8*7, 200), data -> {
            for (int i = 0; i < 8*7; i++)
                for (int j = 0; j < 200; j++)
                    data.setFloat(weights1[i][j], i, j);
        });
        RLAgent.copying = RLAgent.tf.assign(RLAgent.weights1, RLAgent.tf.constant(mrtviTenzor));
        RLAgent.session.runner()
                .addTarget(RLAgent.copying)
                .run();

        mrtviTenzor = TFloat32.tensorOf(Shape.of(200), data -> {
            for (int i = 0; i < 200; i++)
                data.setFloat(biases1[i], i);
        });
        RLAgent.copying = RLAgent.tf.assign(RLAgent.biases1, RLAgent.tf.constant(mrtviTenzor));
        RLAgent.session.runner()
                .addTarget(RLAgent.copying)
                .run();

        mrtviTenzor = TFloat32.tensorOf(Shape.of(200, 150), data -> {
            for (int i = 0; i < 200; i++)
                for (int j = 0; j < 150; j++)
                    data.setFloat(weights2[i][j], i, j);
        });
        RLAgent.copying = RLAgent.tf.assign(RLAgent.weights2, RLAgent.tf.constant(mrtviTenzor));
        RLAgent.session.runner()
                .addTarget(RLAgent.copying)
                .run();

        mrtviTenzor = TFloat32.tensorOf(Shape.of(150), data -> {
            for (int i = 0; i < 150; i++)
                data.setFloat(biases2[i], i);
        });
        RLAgent.copying = RLAgent.tf.assign(RLAgent.biases2, RLAgent.tf.constant(mrtviTenzor));
        RLAgent.session.runner()
                .addTarget(RLAgent.copying)
                .run();

        mrtviTenzor = TFloat32.tensorOf(Shape.of(150, 100), data -> {
            for (int i = 0; i < 150; i++)
                for (int j = 0; j < 100; j++)
                    data.setFloat(weights3[i][j], i, j);
        });
        RLAgent.copying = RLAgent.tf.assign(RLAgent.weights3, RLAgent.tf.constant(mrtviTenzor));
        RLAgent.session.runner()
                .addTarget(RLAgent.copying)
                .run();

        mrtviTenzor = TFloat32.tensorOf(Shape.of(100), data -> {
            for (int i = 0; i < 100; i++)
                data.setFloat(biases3[i], i);
        });
        RLAgent.copying = RLAgent.tf.assign(RLAgent.biases3, RLAgent.tf.constant(mrtviTenzor));
        RLAgent.session.runner()
                .addTarget(RLAgent.copying)
                .run();

        mrtviTenzor = TFloat32.tensorOf(Shape.of(100, 52), data -> {
            for (int i = 0; i < 100; i++)
                for (int j = 0; j < 52; j++)
                    data.setFloat(weights4[i][j], i, j);
        });
        RLAgent.copying = RLAgent.tf.assign(RLAgent.weights4, RLAgent.tf.constant(mrtviTenzor));
        RLAgent.session.runner()
                .addTarget(RLAgent.copying)
                .run();

        mrtviTenzor = TFloat32.tensorOf(Shape.of(52), data -> {
            for (int i = 0; i < 52; i++)
                data.setFloat(biases4[i], i);
        });
        RLAgent.copying = RLAgent.tf.assign(RLAgent.biases4, RLAgent.tf.constant(mrtviTenzor));
        RLAgent.session.runner()
                .addTarget(RLAgent.copying)
                .run();
    }


}
