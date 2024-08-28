package players.portfolio.scripts;

import core.actions.Action;
import core.actors.Actor;
import core.game.GameState;
import players.portfolio.scripts.utils.MilitaryFunc;
import utils.Pair;

public class ConvertHighestHPScr extends BaseScript {

    //This script returns the convert action that targets the enemy unit with highest current HP.


    @Override
    public Pair<Action, Double> process(GameState gs, Actor ac) {
        return new MilitaryFunc().getActionByActorAttr(gs, actions, ac, Feature.HP, true);
    }

}
