import core.game.Game;
import gui.GUI;
import org.json.JSONObject;
import org.tensorflow.Session;
import org.tensorflow.SessionFunction;
import org.tensorflow.Signature;
import org.tensorflow.types.TFloat32;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.SavedModelBundle.*;
import players.RLAgent;
import players.RLAgentTrain;
import players.Rewards;
import utils.file.IO;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;

public class Main {

    public static int[] score = new int[2];
    public static void main(String[] args) throws IOException, AWTException {

        RLAgent.initNN();
        RLAgentTrain.initNN();
        JSONObject conf = new IO().readJSON("training.json");
        if(conf.getBoolean("training")){
            for(int i = 0; i < 10001; i++) {
                Play.start();

                for(int k : RLAgent.rewards.keySet()) {
                    ArrayList<Rewards> list = RLAgent.rewards.get(k);
                    float[] G = new float[list.size()];
                    G[list.size()-1] = list.get(list.size()-1).reward;
                    for (int j = list.size()-2; j >= 0; j--)
                        G[j] = Rewards.gamma * G[j+1] + list.get(j).reward;
                    for(int z = 0; z < list.size(); z++){
                        float[] ac = new float[RLAgent.ActionSpaceSize];
                        ac[list.get(z).index] = 1;
                        RLAgent.session.runner()
                                .feed(RLAgent.stateInput.asOutput(), list.get(z).Gstate)
                                .feed(RLAgent.actions.asOutput(), TFloat32.vectorOf(ac))
                                .feed(RLAgent.rew.asOutput(), TFloat32.scalarOf(G[z])) // Feed rewards during the backward pass
                                .addTarget(RLAgent.minimize)
                                .run();
                    }

                }
                if(i > 0 && i % 250 == 0) {
                    Path saveFolder = Files.createDirectory(Path.of("./model-tanh-0.01-"+i));
                    Signature.Builder input = Signature.builder("input").input("input", RLAgent.stateInput);
                    Signature.Builder output = Signature.builder("output").output("probabilities", RLAgent.actionProbabilities);
                    SavedModelBundle.exporter(saveFolder.toString())
                            .withFunctions(SessionFunction.create(input.build(), RLAgent.session), SessionFunction.create(output.build(), RLAgent.session))
                            .withSession(RLAgent.session)//.withFunctions(SessionFunction.create(input.build(), RLAgent.session), SessionFunction.create(output.build(), RLAgent.session))
                            .export();
                    System.out.println(i);

                }
                if(i % 10 == 0){ for(int sc : Game.score) System.out.println(sc);}
                RLAgent.rewards = new HashMap<>();
                if(i % 50 == 0)
                    copyVariablesFromSourceToTarget();
            }


        }else{
            System.out.println("Nije trening");
        }
    }

    private static void copyVariablesFromSourceToTarget() {
        // Fetch and copy weights1
        RLAgentTrain.session.close();
        RLAgentTrain.graph.close();
        RLAgentTrain.initNN();
        TFloat32 sourceWeights1 = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.weights1)
                .run()
                .get(0);
        RLAgentTrain.copying = RLAgentTrain.tf.assign(RLAgentTrain.weights1, RLAgentTrain.tf.constant(sourceWeights1));
        RLAgentTrain.session.runner()
                .addTarget(RLAgentTrain.copying)
                .run();

        // Fetch and copy biases1
        TFloat32 sourceBiases1 = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.biases1)
                .run()
                .get(0);
        RLAgentTrain.copying = RLAgentTrain.tf.assign(RLAgentTrain.biases1, RLAgentTrain.tf.constant(sourceBiases1));
        RLAgentTrain.session.runner()
                .addTarget(RLAgentTrain.copying)
                .run();

// Fetch and copy weights2
        TFloat32 sourceWeights2 = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.weights2)
                .run()
                .get(0);
        RLAgentTrain.copying = RLAgentTrain.tf.assign(RLAgentTrain.weights2, RLAgentTrain.tf.constant(sourceWeights2));
        RLAgentTrain.session.runner()
                .addTarget(RLAgentTrain.copying)
                .run();

        // Fetch and copy biases2
        TFloat32 sourceBiases2 = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.biases2)
                .run()
                .get(0);
        RLAgentTrain.copying = RLAgentTrain.tf.assign(RLAgentTrain.biases2, RLAgentTrain.tf.constant(sourceBiases2));
        RLAgentTrain.session.runner()
                .addTarget(RLAgentTrain.copying)
                .run();

        // Fetch and copy weights3
        TFloat32 sourceWeights3 = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.weights3)
                .run()
                .get(0);
        RLAgentTrain.copying = RLAgentTrain.tf.assign(RLAgentTrain.weights3, RLAgentTrain.tf.constant(sourceWeights3));
        RLAgentTrain.session.runner()
                .addTarget(RLAgentTrain.copying)
                .run();

        // Fetch and copy biases3
        TFloat32 sourceBiases3 = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.biases3)
                .run()
                .get(0);
        RLAgentTrain.copying = RLAgentTrain.tf.assign(RLAgentTrain.biases3, RLAgentTrain.tf.constant(sourceBiases3));
        RLAgentTrain.session.runner()
                .addTarget(RLAgentTrain.copying)
                .run();

        // Fetch and copy weights4
        TFloat32 sourceWeights4 = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.weights4)
                .run()
                .get(0);
        RLAgentTrain.copying = RLAgentTrain.tf.assign(RLAgentTrain.weights4, RLAgentTrain.tf.constant(sourceWeights4));
        RLAgentTrain.session.runner()
                .addTarget(RLAgentTrain.copying)
                .run();

        // Fetch and copy biases4
        TFloat32 sourceBiases4 = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.biases4)
                .run()
                .get(0);
        RLAgentTrain.copying = RLAgentTrain.tf.assign(RLAgentTrain.biases4, RLAgentTrain.tf.constant(sourceBiases4));
        RLAgentTrain.session.runner()
                .addTarget(RLAgentTrain.copying)
                .run();

    }

    }
