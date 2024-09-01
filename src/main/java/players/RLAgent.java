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
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.FloatNdArray;
import org.tensorflow.ndarray.NdArrays;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;
import utils.ElapsedCpuTimer;
import utils.Pair;
import utils.Vector2d;


import java.util.*;


public class RLAgent extends Agent{
    public static Operand<TFloat32> actionProbabilities;
    public static Graph graph;
    public static Session session;
    public static Ops tf;
    static Placeholder<TFloat32> stateInput;
    private Random m_rnd;

    public RLAgent(long seed)
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
            int rand = m_rnd.nextInt(7 * 7 + 3);//plus 3 for upgrade, recover and disband
            /**Tensor mrtviTenzor = tf.reshape(
                    tf.dtypes.cast(tf.constant(input), TFloat32.class),
                    tf.array(-1L, input.length)
            ).asTensor();
            List<TFloat32> actionProbs = (List<TFloat32>) session.runner()
                    .fetch(actionProbabilities)
                    .feed(stateInput.asOutput(), mrtviTenzor)
                    .run()
                    .get(0);
            System.out.println(actionProbs);**/
            action = outputAction(unID, gs, rand);
            System.out.println(action);
            while(action == null) action = outputAction(unID, gs, m_rnd.nextInt(7 * 7 + 3));
            return action;
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
        System.out.println(ouptutNeuron + "\n");
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
            // jos capture za grad i selo i za kretanje
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
            x = uX + 1;
            y = uY;
            System.out.println("ovde sam pokusavam da hodam" + x + " " + y + "," + uX + " " + uY);
            a.setDestination(new Vector2d(x, y));
            if(a.isFeasible(gs)) return a;
            return null;
        }
    }

    public static void initNN(){
        int numActions = 51, input = 7*7*8;
        graph = new Graph();
        session = new Session(graph);
        tf = Ops.create(graph);
        // Define the input placeholder (state input)
        stateInput = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, input)));

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
        Operand<TFloat32> logits = tf.math.add(tf.linalg.matMul(layer3, weights4), biases4);

        // Apply softmax to get the action probabilities
        actionProbabilities = tf.nn.softmax(logits);
    }

    @Override
    public Agent copy() {
        return null;
    }

}
