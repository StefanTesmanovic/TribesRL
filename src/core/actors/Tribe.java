package core.actors;

import core.TechnologyTree;
import core.TribesConfig;
import core.Types;
import core.actors.units.Unit;
import core.game.Board;
import core.game.GameState;
import utils.Vector2d;
import utils.graph.Graph;

import java.util.ArrayList;
import java.util.Random;

public class Tribe extends Actor{

    //Cities this tribe owns.
    private ArrayList<Integer> citiesID;

    //Capital City ID
    private int capitalID;

    //Type of the tribe
    private Types.TRIBE tribe;

    //Technology progress of this tribe
    private TechnologyTree techTree;

    //Current number of stars (resources) of this tribe.
    private int stars; //TODO: compute this amount at the beginning of each turn.

    //Game result for this player.
    private Types.RESULT winner = Types.RESULT.INCOMPLETE;

    //Score for the tribe.
    private int score = 0;

    //Indicates if the position in the board is visible
    private boolean obsGrid[][];

    //Trade network of this tribe
    private Graph tradeNetwork;

    private  ArrayList<Types.TRIBE> tribesMet;

    private ArrayList<Unit> convertedUnits;


    public Tribe(Types.TRIBE tribe)
    {
        this.tribe = tribe;
        init();
    }

    public Tribe(int tribeID, int cityID, Types.TRIBE tribe) {
        this.tribeId = tribeID;
        citiesID.add(cityID);
        this.tribe = tribe;
        init();
    }

    private void init()
    {
        techTree = new TechnologyTree();
        techTree.doResearch(tribe.getInitialTech());
        citiesID = new ArrayList<>();
        stars = TribesConfig.INITIAL_STARS;
        this.tradeNetwork = new Graph();
        tribesMet = new ArrayList<>();
        convertedUnits = new ArrayList<>();
    }

    public void initObsGrid(int size)
    {
        obsGrid = new boolean[size][size];
    }


    public Tribe copy()
    {
        Tribe tribeCopy = new Tribe(this.tribe);
        tribeCopy.tribeId = this.tribeId;
        tribeCopy.stars = this.stars;
        tribeCopy.winner = this.winner;
        tribeCopy.score = this.score;
        tribeCopy.capitalID = this.capitalID;

        tribeCopy.techTree = this.techTree.copy();
        if (tradeNetwork != null) {
         //   tribeCopy.tradeNetwork = this.tradeNetwork.copy();
        }

        tribeCopy.obsGrid = new boolean[obsGrid.length][obsGrid.length];
        for(int i = 0; i < obsGrid.length; ++i)
            for(int j = 0; j < obsGrid.length; ++j)
                tribeCopy.obsGrid[i][j] = obsGrid[i][j];

        tribeCopy.citiesID = new ArrayList<>();
        for(int cityID : citiesID)
        {
            tribeCopy.citiesID.add(cityID);
        }

        tribeCopy.tribesMet = new ArrayList<>();
        for (Types.TRIBE t:tribesMet) {
            tribeCopy.tribesMet.add(t);
        }

        tribeCopy.convertedUnits = new ArrayList<>();

        for (Unit u: convertedUnits) {
            tribeCopy.convertedUnits.add(u);
        }


        return tribeCopy;
    }


    public void clearView(int x, int y)
    {
        clearView(x, y, 1);
    }

    public void clearView(int x, int y, int range)
    {
        int size = obsGrid.length;
        for(int i = x-range; i <= x+range; ++i)
            for(int j = y-range; j <= y+range; ++j)
            {
                //All these positions should be within my view.
                if(i >= 0 && j >= 0 && i < size && j < size)
                {
                    if( !obsGrid [i][j]) {
                        obsGrid[i][j] = true;
                        this.score +=TribesConfig.CLEAR_VIEW_POINTS;
                    }
                }
            }
    }


    public void addCity(int id) {
        citiesID.add(id);
    }

    public void removeCity(int id){
        for(int i=0; i<citiesID.size(); i++){
            if (citiesID.get(i) == id){
                citiesID.remove(i);
                return;
            }
        }
        System.out.println("Error!! city ID "+ id +" does not belong to this tribe");
    }

    public void setTechTree(TechnologyTree techTree) {this.techTree = techTree;}
    public TechnologyTree getTechTree() {return techTree;}

    public Types.TECHNOLOGY getInitialTechnology(){
        return tribe.getInitialTech();
    }

    public void addScore(int score){
        this.score += score;
    }

    public void subtractScore(int score){
        this.score -= score;
    }

    public ArrayList<Integer> getCitiesID() {
        return citiesID;
    }

    public String getName(){return tribe.getName();}

    public boolean[][] getObsGrid() {return obsGrid;}

    public boolean isVisible(int x, int y)
    {
        return obsGrid[x][y];
    }

    public Types.TRIBE getType(){return tribe;}

    public Types.RESULT getWinner() {return winner;}
    public int getScore() {return score;}

    public int getStars() {
        return stars;
    }

    public void setStars(int stars) {
        this.stars = stars;
    }

    public void addStars(int stars) {this.stars += stars;}

    public void subtractStars(int stars) {this.stars -= stars;}

    public void setCapitalID(int capitalID) {
        this.capitalID = capitalID;
    }

    public int getCapitalID() {
        return capitalID;
    }

    public boolean hasCity(int cityId) {
        return this.citiesID.contains(cityId);
    }

    public void setPosition(int x, int y) {position = null;} //this doesn't make sense
    public Vector2d getPosition()
    {
        return null;
    }

    public ArrayList<Types.TRIBE> getTribesMet(){
        return tribesMet;
    }

    public void meetTribe(GameState gs, int tribeID){

        Tribe[] t = gs.getTribes(); // get tribes from boards

        boolean[] inMetTribes = new boolean[t.length];
        //loop through all tribes
        for(int i = 0; i<t.length; i++){
            // if tribes not in tribes met or tribe is itself then do nothing else add to tribesmet arraylist
            if (t[i].tribe == this.tribesMet.get(i) || t[i].tribeId == tribeID){
                inMetTribes[i] = true;
            }
            if(!inMetTribes[i]){
                tribesMet.add(t[i].tribe); // add to this tribe
                t[i].tribesMet.add(this.tribe); // add to met tribe as well

                //Pick a technology at random from the tribe to learn
                TechnologyTree thisTribeTree = getTechTree();
                TechnologyTree metTribeTree = t[i].getTechTree();
                ArrayList<Types.TECHNOLOGY> techInThisTribe = new ArrayList<>(); //Check which tech in this tribe
                ArrayList<Types.TECHNOLOGY> techInMetTribe = new ArrayList<>(); // Check which tech in met tribe
                //Check which technologies both research trees contain

                for (Types.TECHNOLOGY tech: Types.TECHNOLOGY.values()
                     ) {
                    if (thisTribeTree.isResearched(tech))
                        techInThisTribe.add(tech);
                    if (metTribeTree.isResearched(tech))
                        techInMetTribe.add(tech);
                }
                ArrayList<Types.TECHNOLOGY> potentialTechForThisTribe = new ArrayList<>();
                ArrayList<Types.TECHNOLOGY> potentialTechForMetTribe = new ArrayList<>();

                for (int x = 0; i<techInMetTribe.size(); i++){
                    if(!thisTribeTree.isResearched(techInMetTribe.get(x)))
                        potentialTechForThisTribe.add(techInMetTribe.get(x));
                }

                for (int x = 0; i<techInThisTribe.size(); i++){
                    if(!metTribeTree.isResearched(techInThisTribe.get(x)))
                        potentialTechForMetTribe.add(techInThisTribe.get(x));
                }

                if(potentialTechForThisTribe.size() == 0 || potentialTechForMetTribe.size() == 0)
                    continue;

                Types.TECHNOLOGY techToGet = potentialTechForThisTribe.get(gs.getRandomGenerator().nextInt(potentialTechForThisTribe.size()));
                thisTribeTree.doResearch(techToGet);

                techToGet = potentialTechForMetTribe.get(gs.getRandomGenerator().nextInt(potentialTechForMetTribe.size()));
                metTribeTree.doResearch(techToGet);
;
            }
        }

    }

    public void addConvertedUnit (Unit u){
        convertedUnits.add(u);
    }


    public void updateNetwork(boolean[][] tradeNetwork, int[][] tileCityId, Types.BUILDING[][] buildings)
    {
        //TODO: compute the trade network for this tribe

        //We need to start from the capital. If capital is not owned, there's no trade network
        if(!citiesID.contains(capitalID))
        {

        }

        //HOW-TO: Execute Dijkstra from the capital city to all cities owned by this tribe
        //  - For roads and cities, they're set to True in tradeNetwork
        //  - None of the traversed tiles can be owned by an opponent tribe or there's no route.
        //  - Two ports from this tribe are connected if separated by 0, 1, 2 or 3 WATER tiles (of any type)

        //Also: a connection between two cities only gives population bonus if the connection is completed by
        //the tribe that owns the cities!

    }
}
