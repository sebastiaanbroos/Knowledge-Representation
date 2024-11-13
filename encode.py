import os

# Put sudoku rules into dimacs format
def read_sudoku_puzzles(file_path):
    """
    Reads Sudoku puzzles from a text file.

    Parameters:
        file_path (str): Path to the file containing Sudoku puzzles.

    Returns:
        list: A list of puzzle strings.
    """
    puzzles = []
    with open(file_path, 'r') as f:
        content = f.read()
        # Assuming each puzzle is separated by a newline
        raw_puzzles = content.strip().split('\n')
        for raw_puzzle in raw_puzzles:
            # Remove any whitespace
            puzzle = ''.join(raw_puzzle.split())
            puzzles.append(puzzle)
    return puzzles

def get_puzzle_size(puzzle_line):
    """
    Determines the size of the Sudoku puzzle.

    Parameters:
        puzzle_line (str): A string representing a Sudoku puzzle.

    Returns:
        int: The size of the Sudoku grid (e.g., 9 for 9x9).
    """
    length = len(puzzle_line)
    size = int(length ** 0.5)
    if size * size != length:
        raise ValueError(f"Invalid puzzle length: {length}. Cannot determine square size.")
    return size


def get_rule_file_for_size(size):
    """
    Returns the appropriate rule file path for a given puzzle size.

    Parameters:
        size (int): The size of the Sudoku grid.

    Returns:
        str: Path to the rule file.
    """
    rule_files = {
        4: 'sudoku-rules-4x4.txt',
        9: 'sudoku-rules-9x9.txt',
        16: 'sudoku-rules-16x16.txt',
    }
    if size in rule_files:
        return rule_files[size]
    else:
        raise ValueError(f"No rule file available for size {size}x{size} Sudoku.")

import os

def process_and_encode_puzzles(puzzles, output_dir):
    """
    Processes and encodes a list of Sudoku puzzles into DIMACS format.

    Parameters:
        puzzles (list): List of puzzle strings.
        output_dir (str): Directory to save the .cnf files.
    """
    for idx, puzzle_line in enumerate(puzzles):
        try:
            size = get_puzzle_size(puzzle_line)
            rule_file = get_rule_file_for_size(size)
            output_file = os.path.join(output_dir, f'puzzle_{idx + 1}.cnf')
            encode_sudoku_puzzle_with_rules(puzzle_line, size, rule_file, output_file)
            print(f"Encoded puzzle {idx + 1} and saved to {output_file}")
        except Exception as e:
            print(f"Failed to process puzzle {idx + 1}: {e}")

def generate_char_to_value_mapping(size):
    """
    Generates a mapping from characters to numerical values for Sudoku puzzles.

    Parameters:
        size (int): The size of the Sudoku grid.

    Returns:
        dict: A mapping from character to integer value.
    """
    mapping = {}
    # Map numbers 1-9
    for i in range(1, min(size + 1, 10)):
        mapping[str(i)] = i
    # Map letters A-Z for sizes greater than 9
    if size > 9:
        for i in range(10, size + 1):
            char = chr(55 + i)  # 65 is 'A', so 10 should map to 'A'
            mapping[char] = i
            mapping[char.lower()] = i  # Include lowercase letters
    return mapping





def sudoku_rules(size, rule_file):
    """
    Reads a file with Sudoku rules in CNF format and encodes it for use in DIMACS format.

    Parameters:
        size (int): The size of the Sudoku grid (e.g., 4 for 4x4, 9 for 9x9).
        rule_file (str): Path to the file containing the Sudoku rules in CNF format.

    Returns:
        tuple: (num_variables, num_clauses, clauses) where:
            - num_variables is the number of unique variables.
            - num_clauses is the number of clauses.
            - clauses is a list of lists representing each clause.
    """
    clauses = []

    with open(rule_file, 'r') as f:
        for line in f:
            # Skip any comments or non-clause lines
            if line.startswith('p cnf'):
                _, _, num_variables, num_clauses = line.split()
                num_variables = int(num_variables)
                num_clauses = int(num_clauses)
            elif line.startswith('c'):
                continue
            else:
                # Convert the clause to a list of integers
                clause = list(map(int, line.strip().split()))
                clauses.append(clause)

    return num_variables, num_clauses, clauses

def encode_sudoku_puzzle_with_rules(puzzle_line, size, rule_file, output_file):
    """
    Encodes a Sudoku puzzle along with its rules in DIMACS format and writes to a .cnf file.

    Parameters:
        puzzle_line (str): A string representing a Sudoku puzzle with '.' for empty cells.
        size (int): The size of the Sudoku grid (e.g., 16 for 16x16).
        rule_file (str): Path to the file containing the Sudoku rules in CNF format.
        output_file (str): Path to the output .cnf file.
    """
    # Load Sudoku rules from the file
    num_variables, num_clauses, clauses = sudoku_rules(size, rule_file)

    # Convert initial puzzle clues to DIMACS format
    puzzle_clauses = []

    # Generate mapping for alphanumeric values
    char_to_value = generate_char_to_value_mapping(size)

    # Ensure the puzzle line has the correct length
    if len(puzzle_line) != size * size:
        raise ValueError(f"Puzzle line must have exactly {size * size} characters for a {size}x{size} Sudoku.")

    # Encode each character in the puzzle line as a clause
    max_digit_length = len(str(size))
    factor = 10 ** max_digit_length

    for i, char in enumerate(puzzle_line):
        if char != '.':
            row = (i // size) + 1
            col = (i % size) + 1
            value = char_to_value[char.upper()]

            # Encode the literal
            literal = row * (factor ** 2) + col * factor + value
            puzzle_clauses.append([literal])

        
    # Update the clause count to include the puzzle clues
    total_clauses = num_clauses + len(puzzle_clauses)

    # Prepare the DIMACS output
    dimacs_output = [f"p cnf {num_variables} {total_clauses}"]

    # Add the rules and puzzle clauses to the DIMACS output
    for clause in clauses:
        dimacs_output.append(" ".join(map(str, clause)) + " 0")
    for clause in puzzle_clauses:
        dimacs_output.append(" ".join(map(str, clause)) + " 0")

    # Write the DIMACS output to the specified .cnf file
    with open(output_file, 'w') as f:
        f.write("\n".join(dimacs_output))

def main():
    # List of your Sudoku puzzle files
    sudoku_files = [
        '4x4.txt',
        '16x16.txt',
        '1000 sudokus.txt',
        'damnhard.sdk.txt',
        'top91.sdk.txt',
        'top95.sdk.txt',
        'top100.sdk.txt',
        'top870.sdk.txt',
        'top2365.sdk.txt',
    ]

    # Directory to store individual test sets
    base_output_dir = 'cnf_puzzles'
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Directory to store the combined test set
    combined_output_dir = os.path.join(base_output_dir, 'combined')
    os.makedirs(combined_output_dir, exist_ok=True)

    puzzle_counter = 1  # Counter for combined set filenames

    for sudoku_file in sudoku_files:
        # Create a subdirectory for each individual test set
        individual_output_dir = os.path.join(base_output_dir, os.path.splitext(sudoku_file)[0])
        os.makedirs(individual_output_dir, exist_ok=True)

        print(f"Processing file: {sudoku_file}")
        try:
            # Read puzzles from the file
            puzzles = read_sudoku_puzzles(sudoku_file)
            
            # Process each puzzle and save it in the individual and combined directories
            for idx, puzzle_line in enumerate(puzzles):
                # Generate output file paths
                individual_output_file = os.path.join(individual_output_dir, f'puzzle_{idx + 1}.cnf')
                combined_output_file = os.path.join(combined_output_dir, f'puzzle_{puzzle_counter}.cnf')

                # Determine puzzle size and corresponding rule file
                size = get_puzzle_size(puzzle_line)
                rule_file = get_rule_file_for_size(size)

                # Encode the puzzle with rules and save to both directories
                encode_sudoku_puzzle_with_rules(puzzle_line, size, rule_file, individual_output_file)
                encode_sudoku_puzzle_with_rules(puzzle_line, size, rule_file, combined_output_file)

                print(f"Saved {sudoku_file} puzzle {idx + 1} to individual and combined sets")
                
                # Increment the combined set counter
                puzzle_counter += 1

        except Exception as e:
            print(f"Failed to process file {sudoku_file}: {e}")


if __name__ == '__main__':
    main()
