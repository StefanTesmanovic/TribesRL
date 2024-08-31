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
import utils.ElapsedCpuTimer;
import utils.Pair;
import utils.Vector2d;

import java.util.*;

import static core.game.GameSaver.gameToJSON;

public class RLAgent extends Agent{
    private Random rnd;

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
        HashSet<Integer> units = null;


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
            if (units != null && !units.contains(((UnitAction) a).getUnitId())) {
                units.add((((UnitAction) a).getUnitId()));
            }
        }

        Action action = new EndTurn();
        if(units == null) return action;
        for (int unID : units) {
            double[] input = Input(unID, gs);
            int rand = m_rnd.nextInt(7 * 7 + 3);//plus 3 for upgrade, recover and disband
            action = outputAction(unID, gs, rand);
            return action;
        }

        return action;
    }
    //friendly Hp, friendly attack, enemy HP, enemy attack enemy--> HP is defence succes if attacked attack is how succesful would our attack be
    private static double[] Input(int unitId, GameState gs){
        int Range = 3, x, y;//7 = 3 + 1 + 3
        double[] ret = new double[7*7*8];
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
                        ret[i + enemyHP] = pair.getSecond();
                        ret[i + enemyAttack] = pair.getFirst();
                    } else {
                        temp = board.getUnitAt(x, y);
                        ret[i] = ((Unit)temp).getCurrentHP();
                        ret[i + friendlyAttack] = ((Unit)temp).ATK;
                    }
                }
                temp = gs.getActor(board.getCityIdAt(x, y));

                if(temp != null)
                    if(temp.getTribeId() != tribeID)
                        ret[i + enemyCity] = ((City)temp).getBound();
                    else
                        ret[i + friendlyCity] = ((City)temp).getBound();
                if(board.getResourceAt(x,y) == Types.RESOURCE.RUINS){ret[i + res] = 1;}
                if(board.getTerrainAt(x,y) == Types.TERRAIN.VILLAGE){ret[i + terr] = 1;}

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
            if(x < 0 || y < 0 || x > gs.getBoard().getSize() || y > gs.getBoard().getSize()) return null;
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
            a.setDestination(new Vector2d(x, y));
            if(a.isFeasible(gs)) return a;
            return null;
        }
    }


    @Override
    public Agent copy() {
        return null;
    }

}
