import ast
import csv

# Input and output file paths
input_file = 'EXAM_ANSWERS.txt'
output_file = 'output.csv'

# List to store dictionaries
data_list = []

# Read the text file and convert each line to a dictionary
with open(input_file, 'r') as file:
    for line in file:
        try:
            # Convert the string representation of a dictionary to a dictionary
            data_dict = ast.literal_eval(line)
            data_list.append(data_dict)
        except SyntaxError as e:
            print(f"Error parsing line: {line}\n{e}")

# Write the dictionaries to a CSV file
if data_list:
    # Extract keys from the first dictionary to use as CSV header
    fieldnames = list(data_list[0].keys())

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header
        writer.writeheader()

        # Write data
        writer.writerows(data_list)

    print(f"CSV file '{output_file}' has been created.")
else:
    print("No valid dictionaries found in the input file.")
