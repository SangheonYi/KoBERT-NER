HEADER = '\033[95m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
WRONG = "WRONG"
ALL = "ALL"

def split_tsv(line):
    tokens, lables = line.split('\t')
    return tokens.split(), lables.split()

def check_answer(output_line, solution_line, probs):
    is_wrong = False
    output_tokens, output_labels = split_tsv(output_line)
    solution_tokens, solution_labels = split_tsv(solution_line)
    detected_probs = []
    for i, output_label  in enumerate(output_labels):
        if i >= len(solution_labels):
            output_tokens[i] = f"{WARNING}{output_tokens[i]}{ENDC}"
            output_labels[i] = f"{FAIL}{output_labels[i]}{ENDC}"
            detected_probs.append(probs[i])
            is_wrong = True
        elif output_label != solution_labels[i]:
            output_tokens[i] = f"{WARNING}{output_tokens[i]}{ENDC}"
            output_labels[i] = f"{FAIL}{output_labels[i]}{ENDC}"
            solution_tokens[i] = f"{WARNING}{solution_tokens[i]}{ENDC}"
            solution_labels[i] = f"{OKGREEN}{solution_labels[i]}{ENDC}"
            detected_probs.append(probs[i])
            is_wrong = True
    if is_wrong:
        return True, [f"{' '.join(output_tokens)}\t{' '.join(output_labels)}", f"{' '.join(solution_tokens)}\t{' '.join(solution_labels)}"], detected_probs
    return False, [f"{' '.join(output_tokens)}\t{' '.join(output_labels)}", f"{' '.join(solution_tokens)}\t{' '.join(solution_labels)}"], detected_probs

def answer_diff(output_tsv_form, prob_matrix, mode=WRONG):
    with open('res/answer.tsv', 'r', newline='', encoding='UTF-8') as solution_file:
        output_lines = output_tsv_form
        solution_lines = solution_file.readlines()
        i = 0
        cnt = 0
        print(len(output_lines), len(solution_lines))
        for output_line, solution_line in zip(output_lines, solution_lines):
            i += 1
            probs = prob_matrix[i - 1]
            is_wrong, print_line, probs = check_answer(output_line, solution_line, probs)
            if mode == WRONG and is_wrong:
                cnt += 1
                print(f"{OKBLUE}{i}th✨{ENDC}")
                print(f"predited:{print_line[0]}")
                print(f"answer:\t{print_line[1]}")
            elif mode != WRONG:
                probs = prob_matrix[i - 1]
                print(f"{OKBLUE}{i}th✨{ENDC}")
                print(f"predited:{print_line[0]}")
                print(f"answer:\t{print_line[1]}")
            for prob_i, detected in enumerate(probs):
                print(f"{HEADER}{detected}{ENDC}")
                if prob_i == len(probs) - 1:
                    print('')
        print(f"wrong line count: {cnt}")