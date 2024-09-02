package players;


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
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.framework.optimizers.Adam;
import org.tensorflow.framework.optimizers.Optimizer;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Gradients;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.ReduceMin;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;
import utils.ElapsedCpuTimer;
import utils.Pair;
import utils.Vector2d;

import java.util.*;

import static org.tensorflow.op.core.ReduceSum.keepDims;

public class RLAgentTrain extends Agent{
    public static Operand<TFloat32> logits, probabilities;
    public static Operand<TFloat32> actionProbabilities;
    public static Graph graph;
    //public static Tensor mrtviTenzor;
    public static Ops tf;
    public static Placeholder<TFloat32> stateInput;

    public static int ActionSpaceSize = 52;
    public static Session session;
    private Random m_rnd;
    public static Gradients gradients;
    public static Optimizer optimizer;
    public static Placeholder<TFloat32> actions;
    public static Placeholder<TFloat32> rew;
    public static Op minimize;

    public static HashMap<Integer, ArrayList<Rewards>> rewards = new HashMap<>();

    public RLAgentTrain(long seed)
    {
        super(seed);
        m_rnd = new Random(seed);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect) {
        SimpleAgent temp = new SimpleAgent(seed);
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
            float[] input = Input(unID, gs);
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
                    .fetch(actionProbabilities)
                    .feed(stateInput.asOutput(), mrtviTenzor)
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

            for(int k = 0; k < 52; k++){
                action = outputAction(unID, gs, outputIndexes[k]);
                if(action != null){
                    if(!rewards.containsKey(unID))
                        rewards.put(unID, new ArrayList<Rewards>());
                    ArrayList<Rewards> tmp = rewards.get(unID);
                    tmp.add(new Rewards(outputIndexes[k], (new SimpleAgent(seed)).evalAction(gs, action), mrtviTenzor));
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
            int x = ouptutNeuron/7+uX-3, y = ouptutNeuron%7+uX-3;
            if(x < 0 || y < 0 || x >= gs.getBoard().getSize() || y >= gs.getBoard().getSize()) return null;
            if(board.getResourceAt(x,y) == Types.RESOURCE.RUINS && (new Examine(uId)).isFeasible(gs)) return new Examine(uId);
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
            if(board.getTerrainAt(x,y) == Types.TERRAIN.CITY){
                Capture a = new Capture(uId);
                a.setCaptureType(Types.TERRAIN.CITY);
                a.setTargetCity(board.getCityIdAt(x,y));
                if(a.isFeasible(gs)) return a;
                return null;
            }
            if(board.getTerrainAt(x,y) == Types.TERRAIN.VILLAGE){
                Capture a = new Capture(uId);
                a.setCaptureType(Types.TERRAIN.VILLAGE);
                if(a.isFeasible(gs)) return a;
                return null;
            }

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
        int numActions = ActionSpaceSize, input = 7*7*8;
        graph = new Graph();
        tf = Ops.create(graph);
        // Define the input placeholder (state input)
        stateInput = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, input)));
        actions = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, numActions)));
        rew = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1)));


        // First hidden layer: input -> 300 neurons
        Variable<TFloat32> weights1 = tf.variable(tf.random.truncatedNormal(tf.constant(new long[]{input, 300}), TFloat32.class));
        Variable<TFloat32> biases1 = tf.variable(tf.zeros(tf.constant(new long[]{300}), TFloat32.class));
        Operand<TFloat32> layer1 = tf.nn.relu(tf.math.add(tf.linalg.matMul(stateInput, weights1), biases1));

        // Second hidden layer: 300 -> 150 neurons
        Variable<TFloat32> weights2 = tf.variable(tf.random.truncatedNormal(tf.constant(new long[]{300, 150}), TFloat32.class));
        Variable<TFloat32> biases2 = tf.variable(tf.zeros(tf.constant(new long[]{150}), TFloat32.class));
        Operand<TFloat32> layer2 = tf.nn.relu(tf.math.add(tf.linalg.matMul(layer1, weights2), biases2));

        // Third hidden layer: 150 -> 100 neurons
        Variable<TFloat32> weights3 = tf.variable(tf.random.truncatedNormal(tf.constant(new long[]{150, 100}), TFloat32.class));
        Variable<TFloat32> biases3 = tf.variable(tf.zeros(tf.constant(new long[]{100}), TFloat32.class));
        Operand<TFloat32> layer3 = tf.nn.relu(tf.math.add(tf.linalg.matMul(layer2, weights3), biases3));

        // Output layer: 100 -> Number of actions
        Variable<TFloat32> weights4 = tf.variable(tf.random.truncatedNormal(tf.constant(new long[]{100, numActions}), TFloat32.class));
        Variable<TFloat32> biases4 = tf.variable(tf.zeros(tf.constant(new long[]{numActions}), TFloat32.class));
        logits = tf.math.add(tf.linalg.matMul(layer3, weights4), biases4);
        logits = tf.math.add(logits, tf.reduceMin(logits, tf.constant(1), ReduceMin.keepDims(false)));
        // Apply softmax to get the action probabilities
        actionProbabilities = tf.math.div(logits,tf.reduceSum(logits, tf.array(1), keepDims(false)));//tf.nn.softmax(logits);

        Operand<TFloat32> logProbs = tf.math.log(tf.reduceSum(tf.math.mul(actionProbabilities, actions), tf.constant(1)));
        Operand<TFloat32> loss = tf.math.neg(tf.math.mul(logProbs, rew)); // Multiply by rewards

        Optimizer optimizer = new Adam(graph, 0.001f);//Adam.createAdamMinimize(tf, 0.001f)//.create(tf, 0.001f);
        minimize = optimizer.minimize(loss);

        session = new Session(graph);
    }

    @Override
    public Agent copy() {
        return null;
    }

}
