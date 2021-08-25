from qecstruct import (
    LinearCode, 
    hamming_code, repetition_code, random_regular_code,
    BinaryVector, BinaryMatrix
)
import pytest


def test_hamming_code():
    code = hamming_code()

    assert len(code) == 7
    assert code.dimension() == 4
    assert code.minimal_distance() == 3
    assert code.num_checks() == 3
    assert code.num_generators() == 4

    assert code.syndrome_of(BinaryVector(7, [3])) == BinaryVector(3, [0])
    assert code.has_codeword(BinaryVector(7, [0, 1, 2]))

    same_code = LinearCode(par_mat=BinaryMatrix(
        7,
        [
            [1, 2, 3, 4],
            [1, 2, 5, 6],
            [0, 1, 4, 5],
        ]
    ))

    assert code.has_same_codespace(same_code)
    assert code != same_code


def test_repetition_code():
    code = repetition_code(5)

    assert len(code) == 5
    assert code.dimension() == 1
    assert code.minimal_distance() == 5
    assert code.num_checks() == 4
    assert code.num_generators() == 1

    assert code.syndrome_of(BinaryVector(5, [0, 2])) == BinaryVector(4, [0, 1, 2])
    assert code.has_codeword(BinaryVector(5, [0, 1, 2, 3, 4]))

    same_code = LinearCode(par_mat=BinaryMatrix(
        5,
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
        ]
    ))

    assert code.has_same_codespace(same_code)
    assert code != same_code


code_parameters = [
    (12, 8, 3, 4),
    (10, 6, 3, 5),
    (5, 4, 4, 5),
]

@pytest.mark.parametrize("num_bits, num_checks, bit_deg, check_deg", code_parameters)
def random_code(num_bits, num_checks, bit_deg, check_deg):
    code = random_regular_code(num_bits, num_checks, bit_deg, check_deg)

    assert len(code) == num_bits
    assert code.dimension() >= num_bits - num_checks 
    assert code.num_checks() == num_checks
    assert code.num_generators() == num_bits - num_checks
