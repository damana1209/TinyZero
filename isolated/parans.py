from enum import Enum
import typing


class MathStatus(Enum):
    BAD_FORMAT = 0
    WRONG = 1
    RIGHT = 2
    IDK = 3


def extract_solution(solution_str: str) -> None | str:
    """
    solution_str: only the text returned by the LLM.
    The string had "good" formatting if there is only one \\boxed{X} for some X.
    If the string is good, X is returned, else None (indicating bad formatting)
    """
    DELIMITER = r"\boxed{"
    par_mapping = {"(": ")", "{": "}", "[": "]", "<": ">"}

    # checked that there is only one answer
    if solution_str.rfind(DELIMITER) != solution_str.find(DELIMITER):
        return (None, solution_str.rfind(DELIMITER), solution_str.find(DELIMITER))
    else:
        del_idx = solution_str.find(DELIMITER)
        X = ""
        pars_stack: list = []
        if del_idx != 1:  # there was at least one instance
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
                        pass

                # its just a regular char
                else:
                    pass

                X = X + str(char)

            # if we did not return from finding '}' then the DELIMITER was not properly closed
        return None


test1 = r"""
[36m(main_task pid=1583359)[0m Solution string:  To find the value of \( y \) such that the projection of the vector \(\begin{pmatrix} 2 \\ y \\ -5 \end{pmatrix}\) onto the vector \(\begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}\) is \(\frac{5}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}\), we need to use the formula for the projection of a vector \(\mathbf{a}\) onto a vector \(\mathbf{b}\):
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m \text{proj}_{\mathbf{b}} \mathbf{a} = \frac{\mathbf{a} \cdot \mathbf{b}}{\mathbf{b} \cdot \mathbf{b}} \mathbf{b}
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m Here, \(\mathbf{a} = \begin{pmatrix} 2 \\ y \\ -5 \end{pmatrix}\) and \(\mathbf{b} = \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}\). The projection is given as \(\frac{5}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}\).
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m First, we calculate the dot product \(\mathbf{a} \cdot \mathbf{b}\):
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m \mathbf{a} \cdot \mathbf{b} = 2 \cdot 1 + y \cdot (-2) + (-5) \cdot 1 = 2 - 2y - 5 = -3 - 2y
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m Next, we calculate the dot product \(\mathbf{b} \cdot \mathbf{b}\):
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m \mathbf{b} \cdot \mathbf{b} = 1^2 + (-2)^2 + 1^2 = 1 + 4 + 1 = 6
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m Now, we substitute these into the projection formula:
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m \text{proj}_{\mathbf{b}} \mathbf{a} = \frac{-3 - 2y}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix} = \frac{-3 - 2y}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix} = \frac{-3 - 2y}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix} = \frac{-3 - 2y}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix} = \frac{-3 - 2y}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m We know that this projection is equal to \(\frac{5}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}\). Therefore, we can set up the equation:
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m \frac{-3 - 2y}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix} = \frac{5}{6} \begin{pmatrix} 1 \\ -2 \\ 1 \end{pmatrix}
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m By comparing the components, we get:
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m \frac{-3 - 2y}{6} = \frac{5}{6}
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m Multiplying both sides by 6, we get:
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m -3 - 2y = 5
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m Solving for \( y \):
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m -2y = 5 + 3
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m -2y = 8
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m y = -4
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m Thus, the value of \( y \) is \(\boxed{-4}\).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"""

test2 = r"""
[0m Solution string:  To find the maximum value of the expression \(ab^2c^3d^4\) given that \(a + b + c + d = 10\), we can use the method of Lagrange multipliers or the AM-GM inequality. Here, we will use the AM-GM inequality for simplicity.
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m The AM-GM inequality states that for non-negative real numbers \(x_1, x_2, \ldots, x_n\),
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m \frac{x_1 + x_2 + \cdots + x_n}{n} \geq \sqrt[n]{x_1 x_2 \cdots x_n}
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m with equality if and only if \(x_1 = x_2 = \cdots = x_n\).
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m In our case, we have the expression \(ab^2c^3d^4\) and the constraint \(a + b + c + d = 10\). To maximize \(ab^2c^3d^4\), we can use the AM-GM inequality on the terms \(a, b^2, c^3, d^4\). However, it is more straightforward to use the method of Lagrange multipliers or to test specific values that might satisfy the equality condition of the AM-GM inequality.
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m Let's test the values \(a = 10, b = 0, c = 0, d = 0\). This satisfies the constraint \(a + b + c + d = 10\), but since \(b, c, d\) are positive, we need to find a different approach. Instead, let's consider the values \(a = 1, b = 1, c = 1, d = 7\). This satisfies the constraint \(a + b + c + d = 10\), and we can check if this gives the maximum value of \(ab^2c^3d^4\).
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m Substituting \(a = 1, b = 1, c = 1, d = 7\) into the expression, we get:
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m \[
[36m(main_task pid=1583359)[0m ab^2c^3d^4 = 1 \cdot 1^2 \cdot 1^3 \cdot 7^4 = 1 \cdot 1 \cdot 1 \cdot 2401 = 2401
[36m(main_task pid=1583359)[0m \]
[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m To confirm that this is indeed the maximum, we can use the method of Lagrange multipliers or other optimization techniques, but for the purpose of this problem, we can conclude that the maximum value of \(ab^2c^3d^4\) given the constraint \(a + b + c + d = 10\) is \(\boxed{2401}\).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endo
"""

test3 = r"""

[36m(main_task pid=1583359)[0m 
[36m(main_task pid=1583359)[0m Therefore, the domain of the function is \(\boxed{(2, 3) \cup (3, 11) \cup (11, 98) \cup (98, 998) \cup (998, 102)}\).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"""


print(
    extract_solution(test1),
    extract_solution(test2),
    extract_solution(test3),
)
