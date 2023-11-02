# script to count the number of times that fnpicker solution is different from the first translation
"""
Test program to: Connection started
Test program to: Log terminal: ['[Step 1 / 3] Translating code to cpp code...', 'info']
Test program to: Log terminal: ['[Step 2 / 3] Generating unit tests...', 'info']
Test program to: Log UT: ['']
Test program to: Log terminal: ['[Step 2 / 3] Error occurred while generating unit tests', 'Disabled UTs']
Test program to: Log T: ['']
Test program to: Log terminal: ['[Step 1 / 3] Error occurred while translating code', 'No error message provided, but no translations were returned']
Test program to: Log terminal: ['Unexpected error occurred', 'Error occurred while translating code: No error message provided, but no translations were returned']
Test program to: Log solution: ['']
Test program to: Connection closed
ret from gs:
['']
"""


# Results:
# roughly 50% of time fnpicker solution is different from the first translation
# but why then do we not improve the benchmark score? score is almost the same
# case1: both first translation and fnpicker pass benchmark
#   -> our unittests fail for first solution; we find "better" solution with fnpicker
#   -> our unittests are better; but we dont see it in benchmark
#   -> we make already good solutions better; but dont improve failing solutions?
#   -> try more translation and unittests; maybe higher temperature
# case2: both fail
#   -> our unittests are not good enough; might need more than 5
#   -> we have score of 80%; so this cannot happen often
# case3: mix of both
#   -> sometimes fnpicker chooses a better solution, sometimes not
#   -> would be a pretty big coincidence to end up with same benchmark score


import os
file = "log_eval_vertexai_baseline_2023-10-31-21-01-26.txt"
file = "log_eval_vertexai_gs_2023-10-30-23-04-23.txt"
file = "log_eval_gpt35-baseline-2023-11-01-13-50-34.txt"
file = "log_eval_gpt35-gs-2023-11-01-23-19-22.txt"
# UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 4120-4121: invalid continuation byte
with open(file, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

count_diff = 0
count_total = 0
cur_sol = ""
cur_trans = ""
for line in lines:
    if "Log T: [" in line:
        cur_trans = line.split("Log T: [")[1].strip()
    if "Log solution: [" in line:
        cur_sol = line.split("Log solution: [")[1].strip()
        count_total += 1
        if cur_trans == "":
            print("error: no translation")
            print("solution:", cur_sol)
        if cur_trans != cur_sol:
            count_diff += 1
        
    # reset
    if "ret from gs" in line:
        cur_sol = ""
        cur_trans = ""

print("count_diff:", count_diff)
print("count_total:", count_total)

