#!/usr/bin/env python

import re
import pandas as pd


def gtf_to_df(gtf):
    df = pd.read_csv(gtf, sep='\t', header=None, names=['chrom', 'source', 'feature', 'start', 'end', 'score', 'strand', 'frame', 'attributes'], comment='#')
    return df


def parse_attributes(a, regex='[; ]*(.*?) "(.*?)"'):
    """
    Parse the attributes column of a (GENCODE/RefSeq) GTF file.

    Input:
    * a [str]: the attributes element (column 9 of the GTF file)
    * regex [str]: a regular expression that will be iteratively applied to the attribute string to capture attribute key, val pairs. Default should work for GENCODE/RefSeq
    """
    x = [m.groups() for m in re.finditer(regex, a)]
    return {key: val for key, val in x}
