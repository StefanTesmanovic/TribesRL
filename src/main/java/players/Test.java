package players;

import core.actions.Action;
import core.game.GameState;
import utils.ElapsedCpuTimer;

import java.util.ArrayList;
import java.util.Random;

import java.io.*;

public class Test extends Agent {

    private Random rnd;

    public Test(long seed)
    {
        super(seed);
        rnd = new Random(seed);
    }

    @Override
    public Action act(GameState gs, ElapsedCpuTimer ect)
    {
        try {
            // Create a ProcessBuilder instance and specify the Python script
            ProcessBuilder pb = new ProcessBuilder("python3", "test.py");
            Process process = pb.start();

            // Send data to the Python script through the process's output stream
            OutputStreamWriter writer = new OutputStreamWriter(process.getOutputStream());
            writer.write("Hello from Java\n");
            writer.flush(); // Ensure the data is sent to the Python process
            writer.close(); // Close the writer to indicate no more data will be sent

            // Read the output from the Python script through the process's input stream
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                //System.out.println("Python says: " + line);
                line += "abc";
            }
            reader.close(); // Close the reader

            // Capture the error stream
            /**
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String errorLine;
            while ((errorLine = errorReader.readLine()) != null) {
                System.err.println("Python error: " + errorLine);
            }
            errorReader.close();**/

            // Wait for the process to complete and check the exit code
            //int exitCode = process.waitFor();
            //System.out.println("Python script exited with code: " + exitCode);

        } catch (Exception e) {
            e.printStackTrace();}
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
