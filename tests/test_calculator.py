"""
Tests for the safe calculator tool.

Covers: basic arithmetic, operator precedence, exponentiation, modulo,
division by zero, code injection prevention, and edge cases.
"""

import pytest

from backend.tools.calculator import calculate


class TestBasicArithmetic:
    """Test fundamental arithmetic operations."""

    def test_addition(self):
        assert calculate("2 + 2") == "4"

    def test_subtraction(self):
        assert calculate("10 - 3") == "7"

    def test_multiplication(self):
        assert calculate("45 * 27") == "1215"

    def test_division(self):
        assert calculate("100 / 4") == "25"

    def test_float_division(self):
        assert calculate("7 / 2") == "3.5"

    def test_modulo(self):
        assert calculate("10 % 3") == "1"

    def test_exponentiation_double_star(self):
        assert calculate("2 ** 8") == "256"

    def test_exponentiation_caret(self):
        assert calculate("2 ^ 8") == "256"

    def test_negative_number(self):
        assert calculate("-5 + 3") == "-2"

    def test_unary_plus(self):
        assert calculate("+5") == "5"


class TestOperatorPrecedence:
    """Test that operator precedence is respected."""

    def test_mult_before_add(self):
        assert calculate("2 + 3 * 4") == "14"

    def test_parentheses(self):
        assert calculate("(2 + 3) * 4") == "20"

    def test_nested_parentheses(self):
        assert calculate("((2 + 3) * (4 - 1))") == "15"

    def test_complex_expression(self):
        assert calculate("10 + 2 * 3 - 4 / 2") == "14"


class TestErrorHandling:
    """Test error cases raise ValueError with descriptive messages."""

    def test_division_by_zero(self):
        with pytest.raises(ValueError, match="Division by zero"):
            calculate("10 / 0")

    def test_empty_expression(self):
        with pytest.raises(ValueError):
            calculate("")

    def test_whitespace_only(self):
        with pytest.raises(ValueError):
            calculate("   ")

    def test_syntax_error(self):
        with pytest.raises(ValueError, match="Invalid expression syntax"):
            calculate("2 + + 3 *")

    def test_large_exponent(self):
        with pytest.raises(ValueError, match="exponent too large"):
            calculate("2 ** 100000")


class TestSecurityPrevention:
    """Test that code injection attempts are blocked."""

    def test_import_blocked(self):
        with pytest.raises(ValueError, match="Unsafe expression"):
            calculate("__import__('os')")

    def test_exec_blocked(self):
        with pytest.raises(ValueError, match="Unsafe expression"):
            calculate("exec('print(1)')")

    def test_eval_blocked(self):
        with pytest.raises(ValueError, match="Unsafe expression"):
            calculate("eval('1+1')")

    def test_attribute_access_blocked(self):
        with pytest.raises(ValueError, match="Unsafe expression"):
            calculate("os.system('ls')")

    def test_function_call_blocked(self):
        with pytest.raises(ValueError, match="Unsafe expression"):
            calculate("print(42)")

    def test_string_literal_blocked(self):
        with pytest.raises(ValueError, match="Unsafe expression"):
            calculate("'hello'")

    def test_list_comprehension_blocked(self):
        with pytest.raises(ValueError, match="Unsafe expression"):
            calculate("[x for x in range(10)]")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_number(self):
        assert calculate("42") == "42"

    def test_float_result_as_int(self):
        # 6.0 should display as "6"
        assert calculate("2.0 * 3.0") == "6"

    def test_actual_float_result(self):
        assert calculate("1 / 3") == str(1 / 3)

    def test_whitespace_handling(self):
        assert calculate("  2  +  3  ") == "5"

    def test_zero(self):
        assert calculate("0") == "0"

    def test_negative_result(self):
        assert calculate("3 - 10") == "-7"
