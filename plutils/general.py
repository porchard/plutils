#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd
import pyarrow.parquet
import re
import gzip
import subprocess
import tempfile


def open_maybe_gzipped(filename):
    """
    Modified from atactk (https://github.com/ParkerLab/atactk)

    Open a possibly gzipped file.
    Parameters
    ----------
    filename: str
        The name of the file to open.
    Returns
    -------
    file
        An open file object.
    """
    if filename == '-':
        return sys.stdin
    else:
        with open(filename, 'rb') as test_read:
            byte1, byte2 = ord(test_read.read(1)), ord(test_read.read(1))
            if byte1 == 0x1f and byte2 == 0x8b:
                f = gzip.open(filename, mode='rt')
            else:
                f = open(filename, 'rt')
        return f


def count_lines(f, read_cmd=None):
    if read_cmd is None:
        return int(subprocess.run(['wc', '-l', f], capture_output=True, check=True).stdout.decode().split()[0])
    else:
        strm = subprocess.run([read_cmd, f], capture_output=True, check=True)
        return int(subprocess.run(['wc', '-l'], input=strm.stdout, capture_output=True).stdout.decode())


def bin_with_names(x, bin_edges, bin_edge_names=None, stat_name='x'):
    assert(len(bin_edges) == len(bin_edge_names))
    if bin_edge_names is None:
        bin_edge_names = [str(i) for i in bin_edges]
    bin_to_bin_name = dict(zip(bin_edges, bin_edge_names))
    bin_edges_sorted = list(sorted(bin_edges))
    bin_edge_names_sorted = [bin_to_bin_name[i] for i in bin_edges_sorted]

    xbin = np.digitize(x, bin_edges_sorted)

    bins_to_bin_names = []
    bins_to_bin_names.append('{} < {}'.format(stat_name, bin_edge_names_sorted[0]))
    for i in range(len(bin_edges_sorted) - 1):
        bins_to_bin_names.append('{} <= {} < {}'.format(bin_edge_names_sorted[i], stat_name, bin_edge_names_sorted[i+1]))
    bins_to_bin_names.append('{} >= {}'.format(stat_name, bin_edge_names_sorted[-1]))

    xbin = [bins_to_bin_names[x] for x in xbin]
    xbin = pd.Categorical(xbin, categories=bins_to_bin_names, ordered=True)
    return xbin


def merge_intersecting_sets(set_list):
    """
    Given a list of sets, merge all sets that have any overlapping elements
    e.g.:
    x = set(['a', 'b'])
    y = set(['b', 'c'])
    z = set(['b', 'd'])
    merge_intersecting_sets([x, y, z]) --> [{'a', 'b', 'c', 'd'}]
    
    x = set(['a', 'b'])
    y = set(['b', 'c'])
    z = set(['z', 'd'])
    merge_intersecting_sets([x, y, z]) --> [{'a', 'b', 'c'}, {'d', 'z'}]
    """
    if not isinstance(set_list, list):
        raise TypeError('set_list must be a list')
    for i in set_list:
        if not isinstance(i, set):
            raise TypeError('Each element of set_list must be a set')
    sets = {i: x for i, x in enumerate(set_list)}
    merge = True
    while merge:
        merge = False
        keys = list(sorted(sets.keys()))
        for idx_1, key_1 in enumerate(keys):
            for idx_2, key_2 in enumerate(keys):
                if idx_2 <= idx_1:
                    continue
                if len(sets[key_1].intersection(sets[key_2])) > 0:
                    sets[key_1] = sets[key_1].union(sets[key_2])
                    del sets[key_2]
                    merge = True
                    break
            if merge:
                break
    return [sets[k] for k in sorted(sets.keys())]


def list_hdf_columns(f):
    tmp = pd.read_hdf(f, start=0, stop=1)
    return list(tmp.columns)


def list_hdf_rows(f):
    cols = list_hdf_columns(f)
    tmp = pd.read_hdf(f, columns=cols[0])
    return list(tmp.index)


def list_parquet_columns(f):
    schema = pyarrow.parquet.read_schema(f)
    return schema.names


def list_parquet_rows(f):
    columns = list_parquet_columns(f)
    tmp = pd.read_parquet(f, columns=[columns[0]])
    return tmp.index.to_list()


def read_vcf_header(f):
    header = subprocess.run(['bcftools', 'view', '--header-only', f], capture_output=True, check=True).stdout.decode().split('\n')
    if header[-1] == '':
        header = header[:-1]
    return header


def parse_region(s):
    chrom, start, end = re.match('^(.*)[-:_](\d+)[-:_](\d+)$', s).groups()
    start, end = int(start), int(end)
    return (chrom, start, end)


def tabix(f, region, read_header=True):
    """
    f: file to tabix
    region: region, or list of regions, in format chr:start-end (can use :,-,_ as separators)
    read_header: use the last header line prefixed with '#' as column names in the returned dataframe
    """
    if not isinstance(region, list) and not isinstance(region, str):
        raise TypeError('region must be a list or a str')
    if not isinstance(f, str):
        raise TypeError('f must be a str (path to a file)')
    if not isinstance(read_header, bool):
        raise TypeError('read_header must be a bool')
    if isinstance(region, list):
        regions = []
        for r in region:
            chrom, start, end = parse_region(r)
            start, end = str(start), str(end)
            regions.append([chrom, start, end])
        with tempfile.NamedTemporaryFile() as tmpf:
            pd.DataFrame(regions).to_csv(tmpf.name, sep='\t', index=False, header=False)
            sp = subprocess.run(['tabix', '--regions', tmpf.name, f], capture_output=True, check=True)
    else:
        chrom, start, end = parse_region(region)
        sp = subprocess.run(['tabix', f, f'{chrom}:{start}-{end}'], capture_output=True, check=True)
    txt = sp.stdout.decode().split('\n')
    if txt[-1] == '':
        txt = txt[:-1]
    
    if len(txt) == 0:
        return None

    header = None
    if read_header:
        with gzip.open(f, 'rt') as fh:
            for line in fh:
                if line.startswith('#'):
                    header = line.lstrip('#').rstrip().split('\t')
                else:
                    break
    
    return pd.DataFrame([i.split('\t') for i in txt], columns=header)


def parse_ensembl_id(x):
    """
    See https://m.ensembl.org/info/genome/stable_ids/index.html:
    ENS[species prefix][feature type prefix][a unique eleven digit number]
    """
    FEATURE_TYPE_PREFICES = ['E', 'FM', 'G', 'GT', 'P', 'R', 'T']
    ENSEMBL_RE = '^ENS(.*?)({})(\d+)\.?(.*?)$'.format('|'.join(FEATURE_TYPE_PREFICES))
    species_prefix, feature_type_prefix, digits, version = re.match(ENSEMBL_RE, x).groups()
    return {'species_prefix': species_prefix, 'feature_type_prefix': feature_type_prefix, 'digits': digits, 'version': version}

    
def strip_version_from_ensembl_gene_id(x, keep_par_y=False):
    """
    ENSG00000280767.3 --> ENSG00000280767
    if keep_par_y: ENSG00000280767.3_PAR_Y --> ENSG00000280767.PAR_Y
    if !keep_par_y: ENSG00000280767.3_PAR_Y --> ENSG00000280767
    """
    x = parse_ensembl_id(x)
    if keep_par_y and 'PAR_Y' in x['version']:
        return 'ENS{species_prefix}{feature_type_prefix}{digits}.PAR_Y'.format(**x)
    else:
        return 'ENS{species_prefix}{feature_type_prefix}{digits}'.format(**x)