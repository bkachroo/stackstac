"""
This type stub file was generated by pyright.
"""

import logging

logger = logging.getLogger(__name__)
def dict_merge(dct, merge_dct, add_keys=...):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.

    This version will return a copy of the dictionary and leave the original
    arguments untouched.

    The optional argument ``add_keys``, determines whether keys which are
    present in ``merge_dict`` but not ``dct`` should be included in the
    new dict.

    Args:
        dct (dict) onto which the merge is executed
        merge_dct (dict): dct merged into dct
        add_keys (bool): whether to add new keys

    Returns:
        dict: updated dict
    """
    ...

def download_file(url, filename=..., requester_pays=..., headers=...):
    """ Download a file as filename """
    ...

def mkdirp(path):
    """ Recursively make directory """
    ...

def splitall(path):
    ...

def get_s3_signed_url(url, rtype=..., public=..., requester_pays=..., content_type=...):
    ...

def terminal_calendar(events, cols=...):
    """ Get calendar covering all dates, with provided dates colored and labeled """
    ...
