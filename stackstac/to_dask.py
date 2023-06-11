from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple, Type, Union
import warnings

from affine import Affine
import dask
import dask.array as da
from dask.blockwise import blockwise
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ArraySliceDep
import numpy as np
from rasterio import windows
from rasterio.enums import Resampling

from .raster_spec import Bbox, RasterSpec
from .rio_reader import AutoParallelRioReader, LayeredEnv
from .reader_protocol import Reader

from concurrent.futures import ThreadPoolExecutor, as_completed
from distributed.worker import logger
import os
import subprocess
import uuid
import shlex
import requests

ChunkVal = Union[int, Literal["auto"], str, None]
ChunksParam = Union[ChunkVal, Tuple[ChunkVal, ...], Dict[int, ChunkVal]]
RUN_MODE = os.environ.get('RUN_MODE')
TRACE = (RUN_MODE == 'trace')

def items_to_dask(
    asset_table: np.ndarray,
    spec: RasterSpec,
    chunksize: ChunksParam,
    resampling: Resampling = Resampling.nearest,
    dtype: np.dtype = np.dtype("float64"),
    fill_value: Union[int, float] = np.nan,
    rescale: bool = True,
    reader: Type[Reader] = AutoParallelRioReader,
    gdal_env: Optional[LayeredEnv] = None,
    errors_as_nodata: Tuple[Exception, ...] = (),
) -> da.Array:
    "Create a dask Array from an asset table"
    errors_as_nodata = errors_as_nodata or ()  # be sure it's not None

    if not np.can_cast(fill_value, dtype):
        raise ValueError(
            f"The fill_value {fill_value} is incompatible with the output dtype {dtype}. "
            f"Either use `dtype={np.array(fill_value).dtype.name!r}`, or pick a different `fill_value`."
        )

    chunks = normalize_chunks(chunksize, asset_table.shape + spec.shape, dtype)
    chunks_tb, chunks_yx = chunks[:2], chunks[2:]

    # The overall strategy in this function is to materialize the outer two dimensions (items, assets)
    # as one dask array (the "asset table"), then map a function over it which opens each URL as a `Reader`
    # instance (the "reader table").
    # Then, we use the `ArraySliceDep` `BlockwiseDep` to represent the inner inner two dimensions (y, x), and
    # `Blockwise` to create the cartesian product between them, avoiding materializing that entire graph.
    # Materializing the (items, assets) dimensions is unavoidable: every asset has a distinct URL, so that information
    # has to be included somehow.

    # make URLs into dask array, chunked as requested for the time,band dimensions
    asset_table_dask = da.from_array(
        asset_table,
        chunks=chunks_tb,
        inline_array=True,
        name="asset-table-" + dask.base.tokenize(asset_table),
    )

    # map a function over each chunk that opens that URL as a rasterio dataset
    with dask.annotate(fuse=False):
        # ^ HACK: prevent this layer from fusing to the next `fetch_raster_window` one.
        # This uses the fact that blockwise fusion doesn't happen when the layers' annotations
        # don't match, which may not be behavior we can rely on.
        # (The actual content of the annotation is irrelevant here, just that there is one.)
        reader_table = asset_table_dask.map_blocks(
            asset_table_to_reader_and_window,
            spec,
            resampling,
            dtype,
            fill_value,
            rescale,
            gdal_env,
            errors_as_nodata,
            reader,
            dtype=object,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=da.core.PerformanceWarning)

        name = f"fetch_raster_window-{dask.base.tokenize(reader_table, chunks, dtype, fill_value)}"
        # TODO use `da.blockwise` once it supports `BlockwiseDep`s as arguments
        lyr = blockwise(
            fetch_raster_window,
            name,
            "tbyx",
            reader_table.name,
            "tb",
            ArraySliceDep(chunks_yx),
            "yx",
            dtype,
            None,
            fill_value,
            None,
            numblocks={reader_table.name: reader_table.numblocks},  # ugh
        )
        dsk = HighLevelGraph.from_collections(name, lyr, [reader_table])
        rasters = da.Array(dsk, name, chunks, meta=np.ndarray((), dtype=dtype))

    return rasters


ReaderTableEntry = Optional[Tuple[Reader, windows.Window]]


def asset_table_to_reader_and_window(
    asset_table: np.ndarray,
    spec: RasterSpec,
    resampling: Resampling,
    dtype: np.dtype,
    fill_value: Union[int, float],
    rescale: bool,
    gdal_env: Optional[LayeredEnv],
    errors_as_nodata: Tuple[Exception, ...],
    reader: Type[Reader],
) -> np.ndarray:
    """
    "Open" an asset table by creating a `Reader` for each asset.

    This function converts the asset table (or chunks thereof) into an object array,
    where each element contains a tuple of the `Reader` and `Window` for that asset,
    or None if the element has no URL.
    """
    reader_table = np.empty_like(asset_table, dtype=object)
    for index, asset_entry in np.ndenumerate(asset_table):
        url: str | None = asset_entry["url"]
        if url:
            asset_bounds: Bbox = asset_entry["bounds"]
            asset_window = windows.from_bounds(*asset_bounds, spec.transform)

            entry: ReaderTableEntry = (
                reader(
                    url=url,
                    spec=spec,
                    resampling=resampling,
                    dtype=dtype,
                    fill_value=fill_value,
                    rescale=rescale,
                    gdal_env=gdal_env,
                    errors_as_nodata=errors_as_nodata,
                ),
                asset_window,
            )
            reader_table[index] = entry
    return reader_table

def try_read(index, reader, window):
    if check_hdf4(reader.url):
        reader.url = fileize_link(reader.url)
    try:
        data = reader.read(window)
    except RuntimeError as e:
        logger.warning(str(e))
        return index, np.empty((window.height, window.width)) * np.nan
    return index, data

def check_hdf4(url):
    return '.hdf' in url


def strip_link(subdataset_url, extension='.hdf'):
    end_pos = subdataset_url.find(extension) + 4
    http_pos = subdataset_url.find('http')
    if http_pos != -1:
        return subdataset_url[http_pos:end_pos]
    vsi_pos = subdataset_url.find('/vsi')
    if vsi_pos != -1:
        return subdataset_url[vsi_pos:end_pos]
    s3_pos = subdataset_url.find('s3://')
    if s3_pos != -1:
        return subdataset_url[s3_pos:end_pos]
    return None

def fileize_link(subdataset_url, extension='.hdf', file_only=False):
    partial = subdataset_url.split(extension)[0]
    folder_end = partial.rindex('/') + 1
    if file_only:
        return partial[folder_end:] + extension
    http_pos = subdataset_url.find('http')
    if http_pos != -1:
        return subdataset_url[:http_pos] + subdataset_url[folder_end:]
    vsi_pos = subdataset_url.find('/vsi')
    if vsi_pos != -1:
        return subdataset_url[:vsi_pos] + subdataset_url[folder_end:]
    s3_pos = subdataset_url.find('s3://')
    if s3_pos != -1:
        return subdataset_url[:s3_pos] + subdataset_url[folder_end:]
    return None


def bulk_download(urls):
    # write to tempfile
    if urls[0].startswith('http'):
        tmpfilename = 'curlconfig' + str(uuid.uuid4()) + '.txt'
        urltxt = "\n".join(["url = " + z for z in urls] + ["--remote-name-all"])
        with open(tmpfilename, 'w') as f:
            f.write(urltxt)
        if TRACE:
            logger.warning("Start bulk download")
        process = subprocess.Popen(shlex.split(f"curl --retry 3 -s --parallel -L -b ~/.urs_cookies -c ~/.urs_cookies --netrc --remote-name-all -K {tmpfilename}"))

        if TRACE:
            logger.warning("Finished bulk download")

    if urls[0].startswith('s3'):
        if TRACE:
            logger.warning("Start bulk download")
            log_value = 'info'
        else:
            log_value = 'error'
        if not os.path.isfile('nasas3creds'):
            creds = get_nasa_s3_creds()
            with open('nasas3creds', 'w') as f:
                f.write("\n".join(["[nasa]", f"aws_access_key_id={creds['aws_access_key_id']}",
                                   f"aws_secret_access_key={creds['aws_secret_access_key']}",
                                   f"aws_session_token={creds['aws_session_token']}"]))
        tmpfilename = 's3down' + str(uuid.uuid4()) + '.txt'
        f = open(tmpfilename, 'w')
        f.write('\n'.join([f"cp --source-region='us-west-2' --flatten '{url}' ." for url in urls]))
        f.close()
        process = subprocess.Popen(shlex.split(f"./s5cmd "
                                               f"--log {log_value} "
                                               f"--request-payer=requester "
                                               f"--credentials-file nasas3creds "
                                               f"--profile nasa "
                                               f"run {tmpfilename}"))
    return_code = process.wait()
    if TRACE:
        logger.warning("Finished bulk download")
    os.remove(tmpfilename)
    return return_code

def get_nasa_s3_creds():
    # get username/password from netrc file
    netrc_creds = {}
    with open(os.path.expanduser("~/.netrc")) as f:
        g = f.read().strip().replace('\n', ' ').split(' ')
        for i in range(len(g)//2):
            netrc_creds[g[2*i]] = g[2*i+1]

    # request AWS credentials for direct read access
    url = requests.get(
        "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials",
        allow_redirects=False,
    ).headers["Location"]

    raw_creds = requests.get(
        url, auth=(netrc_creds["login"], netrc_creds["password"])
    ).json()

    return dict(
        aws_access_key_id=raw_creds["accessKeyId"],
        aws_secret_access_key=raw_creds["secretAccessKey"],
        aws_session_token=raw_creds["sessionToken"],
        region_name="us-west-2",
    )

def fetch_raster_window(
    reader_table: np.ndarray,
    slices: Tuple[slice, slice],
    dtype: np.dtype,
    fill_value: Union[int, float],
) -> np.ndarray:
    "Do a spatially-windowed read of raster data from all the Readers in the table."
    assert len(slices) == 2, slices
    current_window = windows.Window.from_slices(*slices)

    assert reader_table.size, f"Empty reader_table: {reader_table.shape=}"
    # Start with an empty output array, using the broadcast trick for even fewer memz.
    # If none of the assets end up actually existing, or overlapping the current window,
    # or containing data, we'll just return this 1-element array that's been broadcast
    # to look like a full-size array.
    output = np.broadcast_to(
        np.array(fill_value, dtype),
        reader_table.shape + (current_window.height, current_window.width),
    )
    # check for HDF4 files, assuming one means all are HDF4, and assets are subdatasets
    # on the same file
    HDF4_FLAG = False
    for index, entry in np.ndenumerate(reader_table):
        if entry:
            HDF4_FLAG = check_hdf4(entry[0].url)
            break

    # download files
    if HDF4_FLAG:
        download_urls = [strip_link(r[0].url) for r in reader_table[:, 0]]
        code = bulk_download(download_urls)
        if code != 0:
            raise RuntimeError(f"Download failed! Return code {code}")

    all_empty: bool = True
    entry: ReaderTableEntry
    if TRACE:
        logger.warning("Start read-threadpool.")
    thread_pool = ThreadPoolExecutor(len(reader_table))
    futures = []
    for index, entry in np.ndenumerate(reader_table):
        if entry:
            reader, asset_window = entry
            # Only read if the window we're fetching actually overlaps with the asset
            if windows.intersect(current_window, asset_window):
                # TODO when the Reader won't be rescaling, support passing `output` to avoid the copy?
                futures.append(thread_pool.submit(try_read, index, reader, current_window))
                if TRACE:
                    logger.warning(f"Submitted request {index}")

    for future in as_completed(futures):
        index, data = future.result()
        if TRACE:
            logger.warning(f"Received request {index}")
        if all_empty:
            # Turn `output` from a broadcast-trick array to a real array, so it's writeable
            if (
                np.isnan(data)
                if np.isnan(fill_value)
                else np.equal(data, fill_value)
            ).all():
                # Unless the data we just read is all empty anyway
                continue
            output = np.array(output)
            all_empty = False

        output[index] = data
    thread_pool.shutdown()

    if HDF4_FLAG:
        downloaded_files = [fileize_link(d, file_only=True) for d in download_urls]
        for f in downloaded_files:
            os.remove(f)
    if TRACE:
        logger.warning("Shutdown pool")
    return output


def normalize_chunks(
    chunks: ChunksParam, shape: Tuple[int, int, int, int], dtype: np.dtype
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """
    Normalize chunks to tuple of tuples, assuming 1D and 2D chunks only apply to spatial coordinates

    If only 1 or 2 chunks are given, assume they're for the ``y, x`` coordinates,
    and that the ``time, band`` coordinates should be chunksize 1.
    """
    # TODO implement our own auto-chunking that makes the time,band coordinates
    # >1 if the spatial chunking would create too many tasks?
    if isinstance(chunks, int):
        chunks = (1, 1, chunks, chunks)
    elif isinstance(chunks, tuple) and len(chunks) == 2:
        chunks = (1, 1) + chunks

    return da.core.normalize_chunks(
        chunks,
        shape,
        dtype=dtype,
        previous_chunks=((1,) * shape[0], (1,) * shape[1], (shape[2],), (shape[3],)),
        # ^ Give dask some hint of the physical layout of the data, so it prefers widening
        # the spatial chunks over bundling together items/assets. This isn't totally accurate.
    )
