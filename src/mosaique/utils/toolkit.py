import os
import numpy as np
import polars as pl
import pathlib
import shutil
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Any, Mapping, TypeVar

from pathos.pools import ProcessPool as Pool
from rich.progress import Progress

import base64


def in_container():
    return "IN_CONTAINER" in os.environ


def map_to_volume(path, data_volume):
    path = str(path).replace("\\", "/")
    path = Path(path)
    data_volume = Path(data_volume)
    rel_path = path.relative_to(path.parents[2])
    return data_volume / rel_path


def parallelize_over_axis(
    func,
    array,
    axis=-1,
    num_workers=1,
    debug=False,
    task_name="Working",
    disable_progress=False,
    **func_kwargs,
):
    """Parallel version of np.apply_along_axis for arrays of any dimension."""
    # Move target axis to last position
    array = np.moveaxis(array, axis, -1)

    # Reshape to 2D: (all_other_dims, target_axis_size)
    orig_shape = array.shape
    array_2d = array.reshape(-1, orig_shape[-1])

    slices = [array_2d[i] for i in range(array_2d.shape[0])]

    # Process slices in parallel
    results = calculate_over_pool(
        func,
        slices,
        num_workers=num_workers,
        debug=debug,
        chunksize=100,
        n_jobs=len(slices),
        task_name=task_name,
        disable_progress=disable_progress,
        **func_kwargs,
    )

    return np.array(results).reshape(orig_shape[:-1] + (-1,))


def calculate_over_pool(
    func,
    objects,
    num_workers=None,
    debug=False,
    chunksize=12,
    console=None,
    n_jobs=None,
    task_name="Working",
    disable_progress=False,
    **func_kwargs,
):
    if debug:
        results = [func(object, **func_kwargs) for object in objects]

    else:
        func_part = partial(func, **func_kwargs)
        results = []
        if n_jobs is None:
            n_jobs = num_workers
        with Progress(
            console=console, transient=True, disable=disable_progress
        ) as progress:
            progress_string = f"{task_name}..."
            task_id = progress.add_task(progress_string, total=n_jobs)
            with Pool(num_workers, maxtasksperchild=4) as pool:
                for result in pool.imap(func_part, objects, chunksize=chunksize):
                    results.append(result)
                    progress.advance(task_id)
    return results


def deep_list(mapping: dict[Any, Any]) -> dict[Any, Any]:
    updated_mapping = mapping.copy()
    for k, v in mapping.items():
        if isinstance(v, Mapping):
            updated_mapping[k] = deep_list(updated_mapping[k])
        else:
            updated_mapping[k] = v if isinstance(v, list) else [v]
    return updated_mapping


def save_as_parquet(by_eeg, dest_path, partition_cols=["feature"]):
    dest_path = pathlib.Path(dest_path)
    shutil.rmtree(dest_path, ignore_errors=True)
    dest_path.mkdir(parents=True, exist_ok=True)
    entropy_iter = (i for i in by_eeg)
    for df_eeg in entropy_iter:
        try:
            df_eeg.to_parquet(
                dest_path,
                partition_cols=partition_cols,
                index=False,
            )
        except AttributeError:
            pass
    pass


T = TypeVar("T", pl.DataFrame, pl.LazyFrame)


def convert_to_cat(df: T, var_to_keep: list[str] = None) -> T:
    """Convert all columns to categorical except those in var_to_keep."""
    var_to_keep = var_to_keep or []

    cols_to_convert = [
        pl.col(col).cast(pl.Categorical) if col not in var_to_keep else pl.col(col)
        for col in df.columns
    ]

    return df.with_columns(cols_to_convert)


def encode_ptid(string: str, key: str) -> bytes:
    """Simple Vigenere cipher

    https://gist.github.com/gowhari/fea9c559f08a310e5cfd62978bc86a1a
    """
    encoded_chars = []
    for i in range(len(string)):
        key_c = key[i % len(key)]
        encoded_c = chr(ord(string[i]) + ord(key_c) % 256)
        encoded_chars.append(encoded_c)
    encoded_string = "".join(encoded_chars)
    encoded_string = encoded_string.encode("latin")
    return base64.urlsafe_b64encode(encoded_string).rstrip(b"=")


def decode_ptid(string: bytes, key: str) -> str:
    string = base64.urlsafe_b64decode(string + b"===")
    string = string.decode("latin")
    encoded_chars = []
    for i in range(len(string)):
        key_c = key[i % len(key)]
        encoded_c = chr((ord(string[i]) - ord(key_c) + 256) % 256)
        encoded_chars.append(encoded_c)
    encoded_string = "".join(encoded_chars)
    return encoded_string
