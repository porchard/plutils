#!/usr/bin/env python
# coding: utf-8


def interpret_flag(f):
    if not isinstance(f, int):
        raise ValueError('f must be an integer')
    binary = bin(f).replace('0b', '').rjust(12, '0')
    x = {interpretation: bit == '1' for interpretation, bit in zip(['is_paired', 'is_proper_pair', 'is_unmapped', 'mate_unmapped', 'is_reverse', 'mate_reverse', 'is_first', 'is_second', 'is_secondary', 'fail_qc', 'is_duplicate', 'is_supplementary'], binary[::-1])}
    return x
