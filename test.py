import sys

# Read input from Java (standard input)
input_data = sys.stdin.read().strip()

# Process the data
output_data = f"Received: {input_data} and responding from Python!"

# Send the result back to Java (standard output)
print(output_data)
