package players;

import core.actions.Action;
import core.game.Board;
import core.game.GameState;
import org.json.JSONObject;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;

import static core.game.GameSaver.gameToJSON;

public class RLAgent extends Agent{
    private Random rnd;

    public RLAgent(long seed)
    {
        super(seed);
        rnd = new Random(seed);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect)
    {

        Board board = gs.getBoard();
        JSONObject gameJSON = gameToJSON(gs, board, seed);

        ArrayList<Action> allActions = gs.getAllAvailableActions();
        int nActions = allActions.size();
        Action toExecute = allActions.get(rnd.nextInt(nActions));
//        System.out.println("[Tribe: " + playerID + "] Tick " +  gs.getTick() + ", num actions: " + nActions + ". Executing " + toExecute);
        return toExecute;
    }

    @Override
    public Agent copy() {
        return null;
    }

}
