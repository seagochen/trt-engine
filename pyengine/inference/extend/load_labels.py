import sys


def read_labels_from_file(file_path):
    labels = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Strip any trailing whitespace, including CRLF or LF
                line = line.strip()

                # Skip empty lines
                if line:
                    labels.append(line)
    except FileNotFoundError:
        print(f"Error: Failed to open file: {file_path}")
        sys.exit(1)

    return labels

