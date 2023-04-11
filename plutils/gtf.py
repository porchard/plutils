#!/usr/bin/env python

import re
import pandas as pd
import numpy as np


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


def gtf_to_tss(gtf, feature_id='gene_id'):
    """
    Given a GTF file, create a BED6-style DataFrame of TSS.

    Input:
    gtf: Path to GTF file
    feature_id: Attribute to use for labeling TSS (usually e.g. gene_id, gene_name, or transcript_id)

    Output:
    pandas DataFrame of TSS (chrom, start, end, feature_id, ., strand)
    """


    df = gtf_to_df(gtf)
    df = df[df.feature=='transcript']
    df['tss'] = np.where(df.strand == '+', df.start, df.end)
    df['tss_start'] = df.tss - 1  # BED indexes from 0
    df['tss_end'] = df.tss
    df['id'] = df.attributes.map(lambda x: parse_attributes(x)[feature_id])
    df['score'] = '.'
    return df[['chrom', 'tss_start', 'tss_end', 'id', 'score', 'strand']].rename(columns=lambda x: x.replace('tss_', ''))


def gtf_to_gene_bodies(gtf, id='gene_id'):
    """
    Given a GTF file, create a BED6-style DataFrame of gene bodies.

    Input:
    gtf: Path to GTF file
    feature_id: Attribute to use for labeling TSS (usually e.g. gene_id or gene_name)

    Output:
    pandas DataFrame of gene bodies (chrom, start, end, feature_id, ., strand)
    """

    df = gtf_to_df(gtf)
    df = df[df.feature=='gene']
    df['id'] = df.attributes.map(lambda x: parse_attributes(x)[id])
    df['start'] = df.start - 1 # BED indexes from 0
    df['score'] = '.'
    return df[['chrom', 'start', 'end', 'id', 'score', 'strand']]
