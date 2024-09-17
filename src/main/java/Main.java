import core.game.Game;
import org.json.JSONObject;
import org.tensorflow.*;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.types.TFloat32;
import players.RLAgent;
import players.RLAgentTrain;
import players.Rewards;
import utils.file.IO;

import java.awt.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Objects;

public class Main {

    public static int[] score = new int[2];
    public static void main(String[] args) throws IOException, AWTException {
        double startTime = System.currentTimeMillis();

        JSONObject conf = new IO().readJSON("training.json");
        if(Objects.equals(conf.getString("runMode"), "Training")){
            RLAgent.initNN();
            RLAgentTrain.initNN();
            for(int i = 0; i < 51; i++) {
                Play.start();

                ArrayList<float[]> inStates = new ArrayList<>();
                ArrayList<float[]> actions = new ArrayList<>();
                ArrayList<Float> rew = new ArrayList<>();

                for(int k : RLAgent.rewards.keySet()) {
                    ArrayList<Rewards> list = RLAgent.rewards.get(k);

                    float[] G = new float[list.size()];
                    G[list.size()-1] = list.getLast().reward;
                    for (int j = list.size()-2; j >= 0; j--)
                        G[j] = Rewards.gamma * G[j + 1] + list.get(j).reward;

                    for(int z = 0; z < list.size(); z++){
                        float[] ac = new float[RLAgent.ActionSpaceSize];
                        ac[list.get(z).index] = 1;
                        inStates.add(list.get(z).Gstate);
                        actions.add(ac);
                        rew.add(G[z]);
                    }

                }
                if(i > 0 && i % 50 == 0) {
                    Path saveFolder = Files.createDirectory(Path.of("./modelTest-"+i));
                    Signature.Builder input = Signature.builder("input").input("input", RLAgent.stateInput.asOutput());
                    Signature.Builder output = Signature.builder("output").output("probabilities", RLAgent.actionProbabilities);
                    SavedModelBundle.exporter(saveFolder.toString())
                            .withFunctions(SessionFunction.create(input.build(), RLAgent.session), SessionFunction.create(output.build(), RLAgent.session))
                            .withSession(RLAgent.session)//.withFunctions(SessionFunction.create(input.build(), RLAgent.session), SessionFunction.create(output.build(), RLAgent.session))
                            .export();
                    System.out.println(i);
                }
                if(i % 10 == 0){ System.out.println(i + "   " +(System.currentTimeMillis()-startTime)/60000); for(int sc : Game.score) System.out.println(sc);}
                if(i>0 && i % 50 == 0)
                    copyVariablesFromSourceToTarget();

                if(inStates.size() == 0 || actions.size() == 0 || rew.size() == 0){ RLAgent.rewards = new HashMap<>();System.out.println("\nkoji kurac, kako ovo uopste da se desi?\n"); continue;}
                TFloat32 inputStates = TFloat32.tensorOf(Shape.of(inStates.size(), inStates.get(0).length), data -> {
                    for (int g = 0; g < inStates.size(); g++)
                        for(int gg = 0; gg < inStates.get(0).length; gg++)
                            data.setFloat( inStates.get(g)[gg], g, gg);
                });

                TFloat32 OHEActions = TFloat32.tensorOf(Shape.of(actions.size(), actions.get(0).length), data -> {
                    for (int g = 0; g < actions.size(); g++)
                        for(int gg = 0; gg < actions.get(0).length; gg++)
                            data.setFloat( actions.get(g)[gg], g, gg);
                });
                TFloat32 rewards = TFloat32.tensorOf(Shape.of(rew.size()), data -> {
                    for (int g = 0; g < rew.size(); g++)
                        data.setFloat(rew.get(g), g);
                });

                RLAgent.session.runner()
                        .feed(RLAgent.stateInput.asOutput(), inputStates)
                        .feed(RLAgent.actions.asOutput(), OHEActions)
                        .feed(RLAgent.rew.asOutput(), rewards)
                        .addTarget(RLAgent.minimize)
                        .run();

                RLAgent.rewards = new HashMap<>();
            }
        }else if(Objects.equals(conf.getString("runMode"), "Testing")){
            RLAgent.initNN();
            TFloat32 mrtviTenzor = TFloat32.tensorOf(Shape.of(1, 8*7), data -> {
                for (int i = 0; i < 8*7; i++) {
                    data.setFloat((float) 1.5, 0, i);
                }
            });
            TFloat32 actionProbs = (TFloat32) RLAgent.session.runner()
                    .feed(RLAgent.stateInput.asOutput(), mrtviTenzor)
                    .fetch(RLAgent.actionProbabilities)
                    .run()
                    .get(0);
            saveModel("./ModelTest");
            loadModel("./ModelTest");
            TFloat32 actionProbs2 = (TFloat32) RLAgent.session.runner()
                    .feed(RLAgent.stateInput.asOutput(), mrtviTenzor)
                    .fetch(RLAgent.actionProbabilities)
                    .run()
                    .get(0);
            for(int k = 0; k < 52; k++)
            System.out.println(actionProbs2.getFloat(0,k) == actionProbs.getFloat(0,k));
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

    public static void saveModel(String filePath) throws IOException {

        float[][] weights1 = new float[8*7][200];
        float[] biases1 = new float[200];
        float[][] weights2 = new float[200][150];
        float[] biases2 = new float[150];
        float[][] weights3 = new float[150][100];
        float[] biases3 = new float[100];
        float[][] weights4 = new float[100][52];
        float[] biases4 = new float[52];

        TFloat32 placeholder;
        placeholder = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.weights1)
                .run()
                .get(0);
        for (int i = 0; i < 8*7; i++)
            for (int j = 0; j < 200; j++)
                weights1[i][j] = placeholder.getFloat(i, j);

        placeholder = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.biases1)
                .run()
                .get(0);
        for (int i = 0; i < 200; i++)
            biases1[i] = placeholder.getFloat(i);

        placeholder = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.weights2)
                .run()
                .get(0);
        for (int i = 0; i < 200; i++)
            for (int j = 0; j < 150; j++)
                weights2[i][j] = placeholder.getFloat(i, j);

// Fetch and assign values for biases2
        placeholder = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.biases2)
                .run()
                .get(0);
        for (int i = 0; i < 150; i++)
            biases2[i] = placeholder.getFloat(i);

        placeholder = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.weights3)
                .run()
                .get(0);
        for (int i = 0; i < 150; i++)
            for (int j = 0; j < 100; j++)
                weights3[i][j] = placeholder.getFloat(i, j);

        placeholder = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.biases3)
                .run()
                .get(0);
        for (int i = 0; i < 100; i++)
            biases3[i] = placeholder.getFloat(i);

        placeholder = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.weights4)
                .run()
                .get(0);
        for (int i = 0; i < 100; i++)
            for (int j = 0; j < 52; j++)
                weights4[i][j] = placeholder.getFloat(i, j);

        placeholder = (TFloat32) RLAgent.session.runner()
                .fetch(RLAgent.biases4)
                .run()
                .get(0);
        for (int i = 0; i < 52; i++)
            biases4[i] = placeholder.getFloat(i);


        FileWriter writer = new FileWriter(filePath);

        // Write weights1 and biases1
        writer.write("weights1:\n");
        for (float[] row : weights1) {
            for (float val : row) {
                writer.write(val + " ");
            }
            writer.write("\n");
        }
        writer.write("biases1:\n");
        for (float val : biases1) {
            writer.write(val + " ");
        }
        writer.write("\n");

        // Write weights2 and biases2
        writer.write("weights2:\n");
        for (float[] row : weights2) {
            for (float val : row) {
                writer.write(val + " ");
            }
            writer.write("\n");
        }
        writer.write("biases2:\n");
        for (float val : biases2) {
            writer.write(val + " ");
        }
        writer.write("\n");

        // Write weights3 and biases3
        writer.write("weights3:\n");
        for (float[] row : weights3) {
            for (float val : row) {
                writer.write(val + " ");
            }
            writer.write("\n");
        }
        writer.write("biases3:\n");
        for (float val : biases3) {
            writer.write(val + " ");
        }
        writer.write("\n");

        // Write weights4 and biases4
        writer.write("weights4:\n");
        for (float[] row : weights4) {
            for (float val : row) {
                writer.write(val + " ");
            }
            writer.write("\n");
        }
        writer.write("biases4:\n");
        for (float val : biases4) {
            writer.write(val + " ");
        }
        writer.write("\n");

        writer.close();
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

