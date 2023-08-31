from typing import List


def _generate_new_col(col_name: str, duplicate_cols_dict: dict):
    """
    Generate column without leading feature set name.
    Args:
        col_name: the target column name
        duplicate_cols_dict: duplicate features dictionary
    Returns list of the new columns
    """
    feature_full_name: List[str] = col_name.split(".")
    if len(feature_full_name) != 2:
        return col_name
    elif len(feature_full_name) == 2 and duplicate_cols_dict[feature_full_name[1]] > 1:
        return col_name
    elif len(feature_full_name) == 2 and duplicate_cols_dict[feature_full_name[1]] == 1:
        return feature_full_name[1]


def normalize_cols(cols) -> List[str]:
    """
    Normalize cols - try to remove leading feature set name from features
    Args:
        cols: list of column
    Return normalized columns
    """
    duplicate_columns_dict = {}
    for col in cols:
        feature_full_name: List[str] = col.split(".")
        if len(feature_full_name) != 2:
            duplicate_columns_dict[col] = 1
        else:
            fs_name = feature_full_name[1]
            duplicate_columns_dict[fs_name] = duplicate_columns_dict.get(fs_name, 0) + 1
    cols = [_generate_new_col(col_name, duplicate_columns_dict) for col_name in cols]
    return cols
