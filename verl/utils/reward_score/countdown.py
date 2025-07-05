import re
import random
import ast
import operator


# Create an enum class with 4 types
# assert False, (
#     "I do not expect to run the countdown file at all. We are currently doing math_idk"
# )


class CountdownStatus:
    BAD_FORMAT = 0
    WRONG = 1
    RIGHT = 2
    IDK = 3
    WRONG_AND_INCONFIDENT = 4
    WRONG_AND_CONFIDENT = 5
    RIGHT_AND_INCONFIDENT = 6
    RIGHT_AND_CONFIDENT = 7


def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    # solution_str = solution_str.split('\n')[-1]

    answer_pattern = r"<answer>(.*?)</answer>"
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer


def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        # Extract all numbers from the equation
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]

        # Check if all numbers in equation are available
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)

        # Each number should be used exactly once
        return numbers_in_eq == available_numbers
    except:
        return False


def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation using eval() with precautions."""
    try:
        # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")

        # Evaluate the equation with restricted globals and locals
        result = eval(equation_str, {"__builtins__": None}, {})
        return result
    except Exception as e:
        return None


def compute_score(
    solution_str, ground_truth, method="strict", format_score=0.1, score=1.0
):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def compute_score_idk(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.1,
    score=1.0,
    idk_reward=0.5,
):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    if equation is not None and equation.strip() == "I don't know":
        print(f"IDK detected, returning idk_reward: {idk_reward}")
        print(f"Solution string: {solution_str}")
        return idk_reward

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            return score
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score


def extract_confidence(solution_str):
    """Extract the confidence from the solution string."""
    # Remove everything before the last "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split("\n")[-1]

    confidence_pattern = r"<confidence>(.*?)</confidence>"
    match = re.finditer(confidence_pattern, solution_str)
    matches = list(match)
    if matches:
        final_confidence = matches[-1].group(1).strip()
    else:
        final_confidence = None
    return final_confidence


def compute_score_idk_and_answer(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.1,
    correct_and_confident=1,
    correct_and_inconfident=0.8,
    incorrect_and_confident=0.1,
    incorrect_and_inconfident=0.3,
):
    assert format_score == incorrect_and_confident
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    confidence = extract_confidence(solution_str=solution_str)
    # if equation is not None and equation.strip() == "I don't know":
    #     print(f"IDK detected, returning idk_reward: {idk_reward}")
    #     print(f"Solution string: {solution_str}")
    #     return idk_reward

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Extracted confidence: {confidence}")
        print(f"Solution string: {solution_str}")

    if equation is None or confidence is None:
        if do_print:
            print(f"No equation/confidence found")
        return 0, CountdownStatus.BAD_FORMAT

    if confidence not in ["Sure", "Not sure"]:
        if do_print:
            print(f"Invalid confidence: {confidence}")
        return format_score, CountdownStatus.BAD_FORMAT
    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation: {equation}")
        return format_score, CountdownStatus.BAD_FORMAT

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score, CountdownStatus.BAD_FORMAT

        if abs(result - target) < 1e-5:  # Account for floating point precision
            if do_print:
                print(f"Correct equation: {equation} = {result}")
            if confidence == "Sure":
                return correct_and_confident, CountdownStatus.RIGHT_AND_CONFIDENT
            else:
                return correct_and_inconfident, CountdownStatus.RIGHT_AND_INCONFIDENT
        else:
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            if confidence == "Sure":
                return incorrect_and_confident, CountdownStatus.WRONG_AND_CONFIDENT
            else:
                return incorrect_and_inconfident, CountdownStatus.WRONG_AND_INCONFIDENT
    except:
        if do_print:
            print(f"Error evaluating equation")
        return incorrect_and_inconfident, CountdownStatus.BAD_FORMAT


running_acc = 0.5
ema_coeff = 0.999


def compute_score_idk_rs(
    solution_str,
    ground_truth,
    method="strict",
    format_score=0.1,
    score=1.0,
    idk_reward=0.7,
):
    """The scoring function for countdown task.

    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer

    Returns:
        score, status
    """
    global running_acc, ema_coeff
    target = ground_truth["target"]
    numbers = ground_truth["numbers"]

    equation = extract_solution(solution_str=solution_str)
    if equation is not None and equation.strip() == "I don't know":
        print(f"IDK detected, returning idk_reward: {min(running_acc, idk_reward)}")
        print(f"Solution string: {solution_str}")
        # ? this line is a little weird to me. On the next line, we are returning reward that is a weighted average of recent success fraction and the idk_reward.
        running_acc = ema_coeff * running_acc + (1 - ema_coeff) * format_score
        return min(running_acc, idk_reward), CountdownStatus.IDK

    do_print = random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Target: {target} | Numbers: {numbers}")
        print(f"Extracted equation: {equation}")
        print(f"Solution string: {solution_str}")

    if equation is None:
        if do_print:
            print(f"No equation found")
        return 0, CountdownStatus.BAD_FORMAT

    # Validate equation uses correct numbers
    if not validate_equation(equation, numbers):
        if do_print:
            print(f"Invalid equation")
        return format_score, CountdownStatus.WRONG

    # Evaluate equation
    try:
        result = evaluate_equation(equation)
        if result is None:
            if do_print:
                print(f"Could not evaluate equation")
            return format_score, CountdownStatus.WRONG

        if abs(result - target) < 1e-5:  # Account for floating point precision
            running_acc = ema_coeff * running_acc + (1 - ema_coeff) * score
            if do_print:
                print(f"Correct equation: {equation} = {result}")
                print(f"Running accuracy: {running_acc}")
            return score, CountdownStatus.RIGHT
        else:
            running_acc = ema_coeff * running_acc + (1 - ema_coeff) * format_score
            if do_print:
                print(f"Wrong result: equation = {result}, target = {target}")
            return format_score, CountdownStatus.WRONG
    except:
        if do_print:
            print(f"Error evaluating equation")
        return format_score, CountdownStatus.WRONG
