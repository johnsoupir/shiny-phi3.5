import json
import sys

def convert_format(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        # Read each line from the input file
        for line in infile:
            data = json.loads(line.strip())

            # Convert each line to the desired format
            converted_data = {
                "text": f"<human>: {data['prompt']}\n<bot>: {data['response'].replace('<thinking>', '').replace('<reflection>', '').replace('<output>', '').strip()}",
                "metadata": {
                    "source": data.get('source', 'unknown')
                }
            }

            # Write the converted line to the output file
            outfile.write(json.dumps(converted_data) + '\n')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_file.jsonl> <output_file.jsonl>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    convert_format(input_file, output_file)

