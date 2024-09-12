import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;
import org.tensorflow.ndarray.Shape;

import java.util.ArrayList;

public class TestClass {
    public static void main(String[] args) {
        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            ArrayList<float[]> floats = new ArrayList<>();
            floats.add(new float[5]);
            System.out.println(floats.get(0).length);

            // Example input data: two 2x3 matrices
            float[][] matrix1Data = new float[][]{
                    {1.0f, 2.0f, 3.0f},
                    {4.0f, 5.0f, 6.0f}
            };

            float[][] matrix2Data = new float[][]{
                    {0f, 0f, 1f},
                    {0f, 0f, 1f}
            };

            // Create TFloat32 tensors using the provided method for dynamic tensor creation
            TFloat32 matrix1 = TFloat32.tensorOf(Shape.of(matrix1Data.length, matrix1Data[0].length), data -> {
                for (int row = 0; row < matrix1Data.length; row++) {
                    for (int col = 0; col < matrix1Data[row].length; col++) {
                        data.setFloat(matrix1Data[row][col], row, col);
                    }
                }
            });

            TFloat32 matrix2 = TFloat32.tensorOf(Shape.of(matrix2Data.length, matrix2Data[0].length), data -> {
                for (int row = 0; row < matrix2Data.length; row++) {
                    for (int col = 0; col < matrix2Data[row].length; col++) {
                        data.setFloat(matrix2Data[row][col], row, col);
                    }
                }
            });

            // Element-wise multiplication and sum along columns to get row-wise dot product
            var elementWiseMul = tf.math.mul(tf.constant(matrix1), tf.constant(matrix2));
            var rowwiseDotProduct = tf.reduceSum(elementWiseMul, tf.constant(1)); // Sum along axis 1 (columns)

            // Create a session to execute the operation
            try (Session session = new Session(graph)) {
                // Run the row-wise dot product operation
                TFloat32 result = (TFloat32) session.runner().fetch(rowwiseDotProduct).run().get(0);

                // Print the result to the console
                System.out.println("Row-wise dot product result (shape: " + result.shape() + "):");
                System.out.println(result.getFloat(0) + " " + result.getFloat(1));

            }
        }
    }
}
