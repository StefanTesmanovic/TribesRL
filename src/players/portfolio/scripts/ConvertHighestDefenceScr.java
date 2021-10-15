package players.portfolio.scripts;

import core.actions.Action;
import core.actors.Actor;
import core.game.GameState;
import players.portfolio.scripts.utils.MilitaryFunc;
import utils.Pair;

public class ConvertHighestDefenceScr extends BaseScript {

    //This script returns the convert action that converts the most defensive enemy unit. We
    //  understand the most defensive unit as the one with the highest DEFENCE value.

    @Override
    public Pair<Action, Double> process(GameState gs, Actor ac) {
        return new MilitaryFunc().getActionByActorAttr(gs, actions, ac, Feature.DEFENCE, true);
    }

}
