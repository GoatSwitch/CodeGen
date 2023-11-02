"""
example:
=========================================================     =========================================================     =========================================================     =========================================================
AREA_SQUARE_CIRCUMSCRIBED_CIRCLE                              AREA_SQUARE_CIRCUMSCRIBED_CIRCLE                              AREA_SQUARE_CIRCUMSCRIBED_CIRCLE                              AREA_SQUARE_CIRCUMSCRIBED_CIRCLE
--                                                            --                                                            --                                                            --
def find_Area ( r ) :                                         int find_Area ( int r ) {                                     #include <iostream> ;                                         success : None
    return ( 2 * r * r )                                        return ( 2 * r * r ) ;                                      #include <vector> ;
                                                              }                                                             #include <math.h> ;
                                                                                                                            using namespace std ;
                                                                                                                            int find_Area ( int r ) {
                                                                                                                              return ( 2 * r * r ) ;
-                                                             -                                                             }-                                                            -
--                                                            --                                                            --                                                            --

=========================================================     =========================================================     =========================================================     =========================================================
FIND_WHETHER_GIVEN_INTEGER_POWER_3_NOT                        FIND_WHETHER_GIVEN_INTEGER_POWER_3_NOT                        FIND_WHETHER_GIVEN_INTEGER_POWER_3_NOT                        FIND_WHETHER_GIVEN_INTEGER_POWER_3_NOT
--                                                            --                                                            --                                                            --
def check ( n ) :                                             bool check ( int n ) {                                        struct check_struct {                                         compilation :
    return 1162261467 % n == 0                                  return 1162261467 % n == 0 ;                                  static bool check ( int n ) {                               FIND_WHETHER_GIVEN_INTEGER_POWER_3_NOT.cpp: In
                                                              }                                                                 return 1162261467 % n == 0 ;                              function ‘int main()’:
                                                                                                                              }                                                           FIND_WHETHER_GIVEN_INTEGER_POWER_3_NOT.cpp:33:12:
                                                                                                                            }                                                             error: ‘f_filled’ was not declared in this scope    33
                                                                                                                            ;                                                             |         if(f_filled(param0[i]) == f_gold(param0[i]))
-                                                             -                                                             -                                                             |            ^~~~~~~~-
--                                                            --                                                            --                                                            --
"""

import os

file_baseline = "/home/mw3155/dump/important/4nameabxof-bison-baseline/hypotheses/hyp0.python_sa-cpp_sa.test_.vizualize.txt"
file_gs = "/home/mw3155/dump/important/ny611he6un-bison-gs/hypotheses/hyp0.python_sa-cpp_sa.test_.vizualize.txt"

def get_dict(file):

    with open(file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # parse function and outcome line by line
    possible_outcomes = ["success :", "compilation :", "error :", "script_not_found :", "failure :", "timeout :"]
    dict_outcomes = {}
    cur_function_name = ""
    cur_function_outcome = ""
    dict_bodys = {}
    cur_body = ""
    for i, line in enumerate(lines):
        # new function starts with "========================================================="
        cur_body += line
        if line.startswith("========================================================="):
            if cur_function_name != "":
                # add to dict
                dict_outcomes[cur_function_name] = cur_function_outcome
                dict_bodys[cur_function_name] = cur_body
                cur_body = ""

            # first word of next line is function name
            cur_function_name = lines[i+1].split(" ")[0].strip()
            # search for outcome in next line
            for outcome in possible_outcomes:
                if outcome in lines[i+3]:
                    cur_function_outcome = outcome
                    break
            else:
                print("error: no outcome found")
                print("function name:", cur_function_name)
                print("line:", lines[i+3])
                raise Exception("error: no outcome found")

    # add last function
    dict_outcomes[cur_function_name] = cur_function_outcome
    dict_bodys[cur_function_name] = cur_body

    from pprint import pprint
    pprint(dict_outcomes)

    # count outcome types
    from collections import Counter
    counter = Counter()
    for outcome in dict_outcomes.values():
        counter[outcome] += 1
    pprint(counter)

    return dict_outcomes, dict_bodys, counter

baseline_outcomes, baseline_bodys, baseline_counter = get_dict(file_baseline)
gs_outcomes, gs_bodys, gs_counter = get_dict(file_gs)

# compare outcomes
# check if success functions are the same
baseline_success = set()
gs_success = set()
for function_name, outcome in baseline_outcomes.items():
    if outcome == "success :":
        baseline_success.add(function_name)

for function_name, outcome in gs_outcomes.items():
    if outcome == "success :":
        gs_success.add(function_name)

print("baseline success:", len(baseline_success))
print("gs success:", len(gs_success))
print("intersection:", len(baseline_success.intersection(gs_success)))
print("difference baseline:", len(baseline_success.difference(gs_success)))
print("difference gs:", len(gs_success.difference(baseline_success)))

# print differences
print("baseline success but gs not:")
for function_name in baseline_success.difference(gs_success):
    print(function_name)

# print all failures from gs
print("gs failures:")
for function_name, outcome in gs_outcomes.items():
    if outcome == "failure :":
        print(function_name)#, gs_bodys[function_name])
        # search and print results "#Results: 7, 10"
        for line in gs_bodys[function_name].split("\n"):
            if "#Results: " in line:
                print(line.split("#Results: ")[1])


# compare overlap of failures from gs and baseline
baseline_failures = set()
gs_failures = set()
for function_name, outcome in baseline_outcomes.items():
    if outcome == "failure :":
        baseline_failures.add(function_name)

for function_name, outcome in gs_outcomes.items():
    if outcome == "failure :":
        gs_failures.add(function_name)

print("baseline failures:", len(baseline_failures))
print("gs failures:", len(gs_failures))
print("intersection:", len(baseline_failures.intersection(gs_failures)))
print("difference baseline:", len(baseline_failures.difference(gs_failures)))
print("difference gs:", len(gs_failures.difference(baseline_failures)))









