
#!/usr/bin/env python3

import sys
import argparse
import math
import time
from collections import defaultdict
import os
import csv

# Read a .cnf file

def read_dimacs_cnf(file_path):
    """
    Reads a DIMACS CNF file and returns the clauses and metadata.

    Parameters:
    - file_path (str): Path to the DIMACS CNF file.

    Returns:
    - num_vars (int): Number of variables in the CNF.
    - num_clauses (int): Number of clauses in the CNF.
    - clauses (list of list of int): A list of clauses, where each clause is a list of integers.
    """
    clauses = []
    num_vars = 0
    num_clauses = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Ignore comments
            if line.startswith('c'):
                continue

            # Process the problem line
            if line.startswith('p cnf'):
                _, _, num_vars, num_clauses = line.split()
                num_vars = int(num_vars)
                num_clauses = int(num_clauses)

            # Process clauses
            else:
                if line:  # Only process non-empty lines
                    clause = list(map(int, line.split()))
                    if clause[-1] == 0:  # Remove the trailing 0 at the end of each clause
                        clause.pop()
                    clauses.append(clause)

    return num_vars, num_clauses, clauses


# Read a .cnf file
def read_dimacs_cnf(file_path):
    """
    Reads a DIMACS CNF file and returns the clauses and metadata.

    Parameters:
    - file_path (str): Path to the DIMACS CNF file.

    Returns:
    - num_vars (int): Number of variables in the CNF.
    - num_clauses (int): Number of clauses in the CNF.
    - clauses (list of list of int): A list of clauses, where each clause is a list of integers.
    """
    clauses = []
    num_vars = 0
    num_clauses = 0

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # Ignore comments
            if line.startswith('c'):
                continue

            # Process the problem line
            if line.startswith('p cnf'):
                _, _, num_vars, num_clauses = line.split()
                num_vars = int(num_vars)
                num_clauses = int(num_clauses)

            # Process clauses
            else:
                if line:  # Only process non-empty lines
                    clause = list(map(int, line.split()))
                    if clause[-1] == 0:  # Remove the trailing 0 at the end of each clause
                        clause.pop()
                    clauses.append(clause)

    return num_vars, num_clauses, clauses

def dpll(clauses, assignments={}, stats=None, heuristic='basic', print_interval=100, depth=0):
    """
    DPLL algorithm to solve a SAT problem given in CNF format.

    Parameters:
        clauses (list of lists of ints): The CNF formula represented as a list of clauses,
                                         where each clause is a list of literals.
        assignments (dict): Current variable assignments (True/False for each variable).
        stats (dict): Statistics for tracking calls, unit propagations, and literal choices.
        heuristic (str): Heuristic to use for choosing literals. Options are 'basic', 'jeroslow-wang', 'shannon-entropy', 'hybrid'.
        print_interval (int): Number of calls between progress print statements.
        depth (int): Current recursion depth.

    Returns:
        dict or None: Returns a satisfying assignment as a dictionary if SAT, otherwise None.
    """
    


    stats['calls'] += 1  # Count recursive calls
    
    # Update maximum recursion depth
    if depth > stats.get('max_depth', 0):
        stats['max_depth'] = depth

    # Simplify the formula based on current assignments
    clauses = simplify_formula(clauses, assignments)

    # If no clauses remain, the formula is satisfied
    if not clauses:
        return assignments

    # If any clause is empty, the formula is unsatisfiable under the current assignment
    if any(len(clause) == 0 for clause in clauses):
        stats['conflicts'] += 1
        return None

    # Apply unit propagation
    unit_clause = find_unit_clause(clauses)
    if unit_clause is not None:
        stats['unit_props'] += 1  # Count unit propagations
        literal = unit_clause[0]
        return dpll(clauses, {**assignments, abs(literal): (literal > 0)}, stats, heuristic, print_interval, depth + 1)

    # Choose a literal using the specified heuristic
    start_time = time.process_time()
    if heuristic == 'basic':
        literal = choose_literal_basic(clauses, assignments)
    elif heuristic == 'jeroslow-wang':
        literal = choose_literal_jeroslow_wang(clauses, assignments)
    elif heuristic == 'shannon-entropy':
        literal = choose_literal_shannon_entropy(clauses, assignments)
    elif heuristic == 'hybrid':
        literal = choose_literal_hybrid(clauses, assignments)
    else:
        raise ValueError("Unknown heuristic")
    heuristic_time = time.process_time() - start_time
    stats.setdefault('heuristic_time', 0)
    stats['heuristic_time'] += heuristic_time

    stats['literal_choices'] += 1  # Count literal choices

    # Recursively try both possible assignments
    result = dpll(clauses, {**assignments, abs(literal): True}, stats, heuristic, print_interval, depth + 1)
    if result is not None:
        return result
    else:
        stats['backtracks'] += 1  # Count backtracks

    result = dpll(clauses, {**assignments, abs(literal): False}, stats, heuristic, print_interval, depth + 1)
    if result is not None:
        return result
    else:
        stats['backtracks'] += 1  # Count backtracks

    return None

def simplify_formula(clauses, assignments):
    """
    Simplifies the formula based on current assignments.

    Parameters:
        clauses (list of lists of ints): The CNF formula as a list of clauses.
        assignments (dict): The current variable assignments.

    Returns:
        list of lists of ints: The simplified formula.
    """
    simplified_clauses = []
    for clause in clauses:
        new_clause = []
        clause_satisfied = False
        for literal in clause:
            var = abs(literal)
            if var in assignments:
                # Check if clause is satisfied
                if (literal > 0 and assignments[var]) or (literal < 0 and not assignments[var]):
                    clause_satisfied = True
                    break  # Skip the entire clause
            else:
                new_clause.append(literal)  # Keep literals with unassigned variables

        if not clause_satisfied:
            if not new_clause:
                return [[]]  # Early return with an empty clause to indicate conflict
            simplified_clauses.append(new_clause)

    return simplified_clauses

def find_unit_clause(clauses):
    """
    Finds a unit clause in the list of clauses.

    Parameters:
        clauses (list of lists of ints): The CNF formula as a list of clauses.

    Returns:
        list or None: A unit clause if one exists, otherwise None.
    """
    for clause in clauses:
        if len(clause) == 1:
            return clause
    return None

def choose_literal_basic(clauses, assignments):
    """
    Chooses the first unassigned literal in the formula.

    Parameters:
        clauses (list of lists of ints): The CNF formula as a list of clauses.
        assignments (dict): The current variable assignments.

    Returns:
        int: A literal to assign in the DPLL algorithm.
    """
    for clause in clauses:
        for literal in clause:
            var = abs(literal)
            if var not in assignments:
                return literal
    return None

def choose_literal_jeroslow_wang(clauses, assignments):
    """
    Implements the Jeroslow-Wang heuristic to choose the best literal.
    """
    literal_scores = defaultdict(float)

    for clause in clauses:
        clause_size = len(clause)
        weight = 2 ** (-clause_size)
        for literal in clause:
            var = abs(literal)
            if var not in assignments:
                literal_scores[literal] += weight

    if not literal_scores:
        return None

    # Select the literal with the highest score
    best_literal = max(literal_scores, key=literal_scores.get)
    return best_literal

def choose_literal_shannon_entropy(clauses, assignments):
    """
    Implements a Shannon entropy-based heuristic to choose the best literal.

    The heuristic selects the variable with the highest entropy, indicating the highest uncertainty.
    It then selects the polarity (positive or negative) based on the higher occurrence in the clauses.

    Parameters:
    - clauses: List of clauses, where each clause is a list of literals.
    - assignments: Dictionary of current variable assignments.

    Returns:
    - best_literal: The literal (positive or negative) with the highest entropy.
                    Returns None if no unassigned variables are left.
    """

    # Step 1: Count occurrences of each literal
    literal_counts = defaultdict(int)
    variable_set = set()

    for clause in clauses:
        for literal in clause:
            var = abs(literal)
            if var not in assignments:
                literal_counts[literal] += 1
                variable_set.add(var)

    if not variable_set:
        return None  # All variables are assigned

    # Step 2: Calculate probabilities and entropy for each variable
    variable_entropy = {}

    for var in variable_set:
        pos_count = literal_counts.get(var, 0)
        neg_count = literal_counts.get(-var, 0)
        total = pos_count + neg_count

        if total == 0:
            entropy = 0
        else:
            p = pos_count / total
            # Handle edge cases where p is 0 or 1
            if p == 0 or p == 1:
                entropy = 0
            else:
                entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)

        variable_entropy[var] = entropy

    # Step 3: Select the variable with the highest entropy
    best_var = max(variable_entropy, key=variable_entropy.get)

    # Step 4: Decide the polarity based on higher occurrence
    pos_count = literal_counts.get(best_var, 0)
    neg_count = literal_counts.get(-best_var, 0)

    if pos_count > neg_count:
        best_literal = best_var
    elif neg_count > pos_count:
        best_literal = -best_var
    else:
        # If counts are equal, default to positive literal
        best_literal = best_var

    return best_literal

def choose_literal_hybrid(clauses, assignments):
    """
    Combines Jeroslow-Wang and Shannon entropy heuristics to choose the best literal.

    Parameters:
        clauses (list of lists of ints): The CNF formula as a list of clauses.
        assignments (dict): The current variable assignments.

    Returns:
        int: The literal to choose next.
    """
    # Step 1: Compute Jeroslow-Wang scores
    jw_scores = defaultdict(float)
    for clause in clauses:
        clause_size = len(clause)
        weight = 2 ** (-clause_size)
        for literal in clause:
            var = abs(literal)
            if var not in assignments:
                jw_scores[literal] += weight

    # Step 2: Compute Shannon entropy scores
    # Count occurrences of each literal
    literal_counts = defaultdict(int)
    variable_set = set()
    for clause in clauses:
        for literal in clause:
            var = abs(literal)
            if var not in assignments:
                literal_counts[literal] += 1
                variable_set.add(var)
    # Calculate entropy for each variable
    entropy_scores = {}
    for var in variable_set:
        pos_count = literal_counts.get(var, 0)
        neg_count = literal_counts.get(-var, 0)
        total = pos_count + neg_count

        if total == 0:
            entropy = 0
        else:
            p = pos_count / total
            # Handle edge cases where p is 0 or 1
            if p == 0 or p == 1:
                entropy = 0
            else:
                entropy = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        entropy_scores[var] = entropy

    # Step 3: Normalize the scores
    # First, get the maximum score in each heuristic to normalize them
    max_jw_score = max(jw_scores.values()) if jw_scores else 1
    max_entropy_score = max(entropy_scores.values()) if entropy_scores else 1

    # Step 4: Combine the scores
    combined_scores = {}
    for var in variable_set:
        # Get the JW scores for both polarities
        jw_score_pos = jw_scores.get(var, 0)
        jw_score_neg = jw_scores.get(-var, 0)
        # Normalize
        jw_score_pos /= max_jw_score
        jw_score_neg /= max_jw_score

        # Get entropy score
        entropy_score = entropy_scores.get(var, 0)
        entropy_score /= max_entropy_score

        # Combine scores for both literals
        combined_score_pos = jw_score_pos + entropy_score
        combined_score_neg = jw_score_neg + entropy_score

        # Store in combined_scores
        combined_scores[var] = {'pos': combined_score_pos, 'neg': combined_score_neg}

    # Step 5: Select the literal with the highest combined score
    best_literal = None
    best_score = -1
    for var in variable_set:
        if combined_scores[var]['pos'] > best_score:
            best_score = combined_scores[var]['pos']
            best_literal = var
        if combined_scores[var]['neg'] > best_score:
            best_score = combined_scores[var]['neg']
            best_literal = -var

    return best_literal

def main():
    parser = argparse.ArgumentParser(description='SAT Solver using DPLL algorithm.')
    parser.add_argument('-S', '--strategy', required=True, help='Strategy number (1: basic, 2: Jeroslow-Wang, 3: Shannon entropy, 4: Hybrid)')
    parser.add_argument('inputfile', help='Input file containing the CNF formula in DIMACS format')

    args = parser.parse_args()

    # Map strategy numbers to heuristic names
    strategy_map = {
        '1': 'basic',
        '2': 'jeroslow-wang',
        '3': 'shannon-entropy',
        '4': 'hybrid'
    }

    heuristic = strategy_map.get(args.strategy)
    if heuristic is None:
        print('Invalid strategy number. Choose from 1 (basic), 2 (Jeroslow-Wang), 3 (Shannon entropy), 4 (Hybrid)')
        sys.exit(1)

    # Read the CNF formula from the input file
    num_vars, num_clauses, clauses = read_dimacs_cnf(args.inputfile)

    # Initialize stats dictionary with default values
    stats = {
        'calls': 0,
        'unit_props': 0,
        'literal_choices': 0,
        'max_depth': 0,
        'backtracks': 0,
        'conflicts': 0,
        'heuristic_time': 0
    }

    # Record start time
    start_time = time.time()

    # Solve the CNF formula using the selected heuristic
    result = dpll(clauses, stats=stats, heuristic=heuristic)

    # Record end time
    end_time = time.time()
    total_time = end_time - start_time

    # Output results and statistics
    if result is not None:
        print('SATISFIABLE')
        assignment_list = [var if result.get(var, False) else -var for var in range(1, num_vars + 1)]
        #print('v', ' '.join(map(str, assignment_list)), '0')
    else:
        print('UNSATISFIABLE')

    print("\nStatistics:")
    print(f"Total Execution Time: {total_time:.4f} seconds")
    print(f"Total Recursive Calls: {stats['calls']}")
    print(f"Number of Unit Propagations: {stats['unit_props']}")
    print(f"Number of Literal Choices: {stats['literal_choices']}")
    print(f"Maximum Recursion Depth: {stats['max_depth']}")
    print(f"Number of Backtracks: {stats['backtracks']}")
    print(f"Number of Conflicts: {stats['conflicts']}")
    print(f"Time Spent in Heuristic Function: {stats['heuristic_time']:.4f} seconds")


if __name__ == '__main__':
    main()
