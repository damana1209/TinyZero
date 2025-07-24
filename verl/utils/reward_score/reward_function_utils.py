def to_unit_interval(s: str | None) -> float | None:
    """
    Attempts to parse a string as a float in the unit interval [0, 1].
    Returns the float if successful and in [0, 1], else returns None.
    """
    if s is None:
        return None
    try:
        val = float(s.strip())
        if 0.0 <= val <= 1.0:
            return val
        else:
            return None
    except Exception:
        return None


def extract_solution_tags(solution_str: str, tag_name: str) -> tuple[str | None, int]:
    """
    Extract text between <tag_name> and </tag_name> tags.

    Args:
        solution_str: The string to search in
        tag_name: The name of the tag (without angle brackets)

    Returns:
        tuple: (extracted_text, closing_tag_end_index) if exactly one pair exists,
               (None, 0) otherwise
    """
    opening_tag = f"<{tag_name}>"
    closing_tag = f"</{tag_name}>"

    # Count occurrences of opening and closing tags
    opening_count = solution_str.count(opening_tag)
    closing_count = solution_str.count(closing_tag)

    # Check if there's exactly one pair
    if opening_count != 1 or closing_count != 1:
        return None, 0

    opening_pos = solution_str.find(opening_tag)
    closing_pos = solution_str.find(closing_tag)

    # Make sure opening comes before closing
    if opening_pos >= closing_pos:
        return None, 0

    # Extract text between tags
    start_content = opening_pos + len(opening_tag)
    end_content = closing_pos
    content = solution_str[start_content:end_content]

    # Find index of > in </tag_name> (position of > character in closing tag)
    closing_tag_gt_index = (
        closing_pos + len(closing_tag) - 1
    )  # ?i think -1 will be useful if its the last character of the string and we don't care to start from >

    return content, closing_tag_gt_index


def extract_solution_square_brackets(
    solution_str: str, DELIMITER=r"\boxed{"
) -> None | str:
    """
    !delimeter should be passed in as an r-string, e.g. `DELIMITER = r"\boxed{"` (expecting curly braces)

    solution_str: only the text returned by the LLM. ASSUMING IT IS RAW -- i.e. `\b` is the charecters `\` and `b` not backspace (this seems to be done by hf tokenizer's decode function by default)
    The string had "good" formatting if there is only one \\boxed{X} for some X.
    If the string has good formatting, X is returned, else None (indicating bad formatting)
    """
    par_mapping = {"(": ")", "{": "}", "[": "]", "<": ">"}

    # checked that there is only one answer
    if solution_str.rfind(DELIMITER) != solution_str.find(DELIMITER):
        return None

    else:
        del_idx = solution_str.find(DELIMITER)
        X = ""
        pars_stack: list = []

        if del_idx != -1:  # there was at least one instance
            for char in solution_str[del_idx + len(DELIMITER) :]:
                # is char open par?
                if char in par_mapping.keys():
                    pars_stack = pars_stack + [
                        char
                    ]  # push the open par to the top of the stack
                # is char close par?
                elif char in par_mapping.values():
                    # closing the first '{'
                    if len(pars_stack) == 0 and char == "}":
                        return X

                    # not closing any valid opening
                    elif len(pars_stack) == 0:
                        return None

                    # closing a valid par (pars_stack has 1 element due to the above check)
                    elif par_mapping[pars_stack[-1]] == char:
                        pars_stack.pop()

                # its just a regular char
                else:
                    pass

                X = X + str(char)

            # if we did not return from finding '}' then the DELIMITER was not properly closed
        # if we did not have 1 del_idx or we existed
        return None


def is_equiv(str1: str | None, str2: str | None, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string
