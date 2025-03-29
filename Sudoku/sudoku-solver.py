from pulp import *

VALS = ROWS = COLS = range(1,10)

Boxes = [
    [(3*i+k+1,3*j+l+1) for k in range(3) for l in range(3)]
    for i in range(3)
    for j in range(3)
]


prob = LpProblem("Sudoku Solver")

choices = LpVariable.dicts("Choice", (VALS, ROWS, COLS), cat="Binary")

for r in ROWS:
    for c in COLS:
        prob += lpSum([choices[v][r][c] for v in VALS]) == 1

for v in VALS:
    for r in ROWS:
        prob += lpSum([choices[v][r][c] for c in COLS]) == 1
    
    for c in COLS:
        prob += lpSum([choices[v][r][c] for r in ROWS]) == 1
    
    for b in Boxes:
        prob += lpSum([choices[v][r][c] for (r,c) in b]) == 1

input_data = [
(8, 1, 1),
(3, 2, 3),
(6, 2, 4),
(7, 3, 2),
(9, 3, 5),
(2, 3, 7),
(5, 4, 2),
(7, 4, 6),
(4, 5, 5),
(5, 5, 6),
(7, 5, 7),
(1, 6, 4),
(3, 6, 8),
(1, 7, 3),
(6, 7, 8),
(8, 7, 9),
(8, 8, 3),
(5, 8, 4),
(1, 8, 8),
(9, 9, 2),
(4, 9, 7)
]

for v, r, c in input_data:
    prob += choices[v][r][c] == 1


prob.writeLP("Sudoku.lp")


sudokuout = open("sudokuout.txt", "w")
while True:
    prob.solve()
    # The status of the solution is printed to the screen
    print("Status:", LpStatus[prob.status])
    # The solution is printed if it was deemed "optimal" i.e met the constraints
    if LpStatus[prob.status] == "Optimal":
        # The solution is written to the sudokuout.txt file
        for r in ROWS:
            if r in [1, 4, 7]:
                sudokuout.write("+-------+-------+-------+\n")
            for c in COLS:
                for v in VALS:
                    if value(choices[v][r][c]) == 1:
                        if c in [1, 4, 7]:
                            sudokuout.write("| ")
                        sudokuout.write(str(v) + " ")
                        if c == 9:
                            sudokuout.write("|\n")
        sudokuout.write("+-------+-------+-------+\n\n")
        # The constraint is added that the same solution cannot be returned again
        prob += (
            lpSum(
                [
                    choices[v][r][c]
                    for v in VALS
                    for r in ROWS
                    for c in COLS
                    if value(choices[v][r][c]) == 1
                ]
            )
            <= 80
        )
    # If a new optimal solution cannot be found, we end the program
    else:
        break
sudokuout.close()