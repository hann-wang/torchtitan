# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Essentially re-written in entirety

import os
import struct
import time
from abc import ABC, abstractmethod
from enum import Enum
from functools import lru_cache
from itertools import accumulate
from types import TracebackType
from typing import Optional, Union

import numpy
import torch

from torchtitan.tools.logging import logger

_INDEX_HEADER = b"MMIDIDX\x00\x00"


def get_idx_path(path_prefix: str) -> str:
    """Get the path to the index file from the prefix

    Args:
        path_prefix (str): The prefix

    Returns:
        str: The path to the index file
    """
    return path_prefix + ".idx"


def get_bin_path(path_prefix: str) -> str:
    """Get the path to the data file from the prefix

    Args:
        path_prefix (str): The prefix

    Returns:
        str: The path to the data file
    """
    return path_prefix + ".bin"


class DType(Enum):
    """The NumPy data type Enum for writing/reading the IndexedDataset indices"""

    uint8 = 1
    int8 = 2
    int16 = 3
    int32 = 4
    int64 = 5
    float64 = 6
    float32 = 7
    uint16 = 8

    @classmethod
    def code_from_dtype(cls, value: type[numpy.number]) -> int:
        """Get the code from the dtype

        Args:
            value (type[numpy.number]): The dtype

        Returns:
            int: The code
        """
        return cls[value.__name__].value

    @classmethod
    def dtype_from_code(cls, value: int) -> type[numpy.number]:
        """Get the dtype from the code

        Args:
            value (int): The code

        Returns:
            type[numpy.number]: The dtype
        """
        return getattr(numpy, cls(value).name)

    @staticmethod
    def size(key: Union[int, type[numpy.number]]) -> int:
        """Get the size of the dtype/code in bytes

        Args:
            key (Union[int, type[numpy.number]]): The dtype or code

        Raises:
            ValueError: If the key is neither dtype nor integer code

        Returns:
            int: The size of the dtype/code in in bytes
        """
        if isinstance(key, int):
            return DType.dtype_from_code(key)().itemsize
        elif numpy.number in key.__mro__:
            return key().itemsize
        else:
            raise ValueError

    @staticmethod
    def optimal_dtype(cardinality: Optional[int]) -> type[numpy.number]:
        """Get the dtype to use for an index of a certain cardinality

        Args:
            cardinality (Optional[int]): The number of elements to be indexed

        Returns:
            type[numpy.number]: The dtype to use for the index
        """
        if cardinality is not None and cardinality < 65500:
            return numpy.uint16
        else:
            return numpy.int32


class _IndexWriter(object):
    """Object class to write the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        dtype (type[numpy.number]): The dtype of the index file
    """

    def __init__(self, idx_path: str, dtype: type[numpy.number]) -> None:
        self.idx_path = idx_path
        self.dtype = dtype

    def __enter__(self) -> "_IndexWriter":
        """Enter the context introduced by the 'with' keyword

        Returns:
            _IndexWriter: The instance
        """
        self.idx_writer = open(self.idx_path, "wb")
        # fixed, vestigial practice
        self.idx_writer.write(_INDEX_HEADER)
        # fixed, vestigial practice
        self.idx_writer.write(struct.pack("<Q", 1))
        # the numeric code for the dtype
        self.idx_writer.write(struct.pack("<B", DType.code_from_dtype(self.dtype)))
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> Optional[bool]:
        """Exit the context introduced by the 'with' keyword

        Args:
            exc_type (Optional[type[BaseException]]): Exception type

            exc_val (Optional[BaseException]): Exception value

            exc_tb (Optional[TracebackType]): Exception traceback object

        Returns:
            Optional[bool]: Whether to silence the exception
        """
        self.idx_writer.close()
        return None

    def write(
        self,
        sequence_lengths: list[Union[int, numpy.integer]],
        sequence_modes: Optional[list[Union[int, numpy.integer]]],
        document_indices: list[Union[int, numpy.integer]],
    ) -> None:
        """Write the index (.idx) file

        Args:
            sequence_lengths (list[int]): The length of each sequence

            sequence_modes (Optional[list[int]]): The mode of each sequences

            document_indices (list[int]): The seqyebce indices demarcating the end of each document
        """
        sequence_pointers = self._sequence_pointers(sequence_lengths)

        # the number of sequences in the dataset
        sequence_count = len(sequence_lengths)
        self.idx_writer.write(struct.pack("<Q", sequence_count))

        # the number of documents in the dataset
        document_count = len(document_indices)
        self.idx_writer.write(struct.pack("<Q", document_count))

        # the number of tokens per sequence
        self.idx_writer.write(
            numpy.array(sequence_lengths, dtype=numpy.int32).tobytes(order="C")
        )

        # the byte offsets for all sequences
        self.idx_writer.write(
            numpy.array(sequence_pointers, dtype=numpy.int64).tobytes(order="C")
        )

        # the sequence indices marking the end of each document
        self.idx_writer.write(
            numpy.array(document_indices, dtype=numpy.int64).tobytes(order="C")
        )

        # the mode per sequence
        if sequence_modes is not None:
            self.idx_writer.write(
                numpy.array(sequence_modes, dtype=numpy.int8).tobytes(order="C")
            )

    def _sequence_pointers(
        self, sequence_lengths: list[Union[int, numpy.integer]]
    ) -> list[int]:
        """Build the sequence pointers per the sequence lengths and dtype size

        Args:
            sequence_lengths (list[int]): The length of each sequence

        Returns:
            list[int]: The pointer to the beginning of each sequence
        """
        itemsize = numpy.int64(DType.size(self.dtype))
        curr_ptr = numpy.int64(0)
        list_ptr = []
        for length in sequence_lengths:
            list_ptr.append(curr_ptr.item())
            curr_ptr += length * itemsize
        return list_ptr


class _IndexReader(object):
    """Object class to read the index (.idx) file

    Args:
        idx_path (str): The path to the index file

        multimodal (bool): Whether the dataset is multimodal
    """

    def __init__(self, idx_path: str, multimodal: bool) -> None:
        logger.info(f"Load the {type(self).__name__} from {idx_path}")

        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            code = struct.unpack("<B", stream.read(1))[0]
            self.dtype = DType.dtype_from_code(code)
            self.dtype_size = DType.size(self.dtype)

            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]

            offset = stream.tell()

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        logger.info("\tExtract the sequence lengths")
        t_beg = time.time()
        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )
        t_end = time.time()
        logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        logger.info("\tExtract the sequence pointers")
        t_beg = time.time()
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        t_end = time.time()
        logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        logger.info("\tExtract the document indices")
        t_beg = time.time()
        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset
            + self.sequence_lengths.nbytes
            + self.sequence_pointers.nbytes,
        )
        t_end = time.time()
        logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        self.sequence_modes = None
        if multimodal:
            logger.info("\tExtract the sequence modes")
            t_beg = time.time()
            self.sequence_modes = numpy.frombuffer(
                self.bin_buffer,
                dtype=numpy.int8,
                count=self.sequence_count,
                offset=offset
                + self.sequence_lengths.nbytes
                + self.sequence_pointers.nbytes
                + self.document_indices.nbytes,
            )
            t_end = time.time()
            logger.debug(f"\t> time elapsed: {t_end - t_beg:4f} seconds")

        assert self.sequence_lengths.shape[0] == len(self)
        assert self.sequence_lengths.shape[0] == self.sequence_count
        assert self.sequence_lengths.shape[0] == self.document_indices[-1]

        logger.info(f"> total number of sequences: {len(self)}")
        logger.info(
            f"> total number of documents: {self.document_indices.shape[0] - 1}"
        )

    def __del__(self) -> None:
        """Clean up the object"""
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()  # type: ignore[attr-defined]
            del self.bin_buffer_mmap

    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: The length of the dataset
        """
        return self.sequence_count

    @lru_cache(maxsize=8)  # noqa: B019
    def __getitem__(
        self, idx: int
    ) -> tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]:
        """Return the pointer, length, and mode at the index

        Args:
            idx (int): The index into the dataset

        Returns:
            tuple[numpy.int32, numpy.int64, Optional[numpy.int8]]: The pointer, length and mode
                at the index
        """
        return (
            self.sequence_pointers[idx],
            self.sequence_lengths[idx],
            self.sequence_modes[idx] if self.sequence_modes is not None else None,
        )


class _BinReader(ABC):
    """Abstract class to read the data (.bin) file"""

    @abstractmethod
    def read(self, dtype: type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from
                reading bytes from the data file starting at `offset`.
        """
        pass


class _MMapBinReader(_BinReader):
    """A _BinReader that memory maps the data (.bin) file

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_file_reader = open(bin_path, mode="rb")
        self._bin_buffer_mmap = numpy.memmap(self._bin_file_reader, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap.data)

    def read(self, dtype: type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from
                reading bytes from the data file starting at `offset`.
        """
        return numpy.frombuffer(
            self._bin_buffer, dtype=dtype, count=count, offset=offset
        )

    def __del__(self) -> None:
        """Clean up the object."""
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()  # type: ignore[attr-defined]
        if self._bin_file_reader is not None:
            self._bin_file_reader.close()
        del self._bin_buffer_mmap
        del self._bin_file_reader


class _FileBinReader(_BinReader):
    """A _BinReader that reads from the data (.bin) file using a file pointer

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_path = bin_path

    def read(self, dtype: type[numpy.number], count: int, offset: int) -> numpy.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from
                reading bytes from the data file starting at `offset`.
        """
        sequence = numpy.empty(count, dtype=dtype)
        with open(self._bin_path, mode="rb", buffering=0) as bin_buffer_file:
            bin_buffer_file.seek(offset)
            bin_buffer_file.readinto(sequence)
        return sequence


class IndexedDataset(torch.utils.data.Dataset):
    """The low-level interface dataset class

    Args:
        path_prefix (str): The index (.idx) and data (.bin) prefix

        multimodal (bool): Whether the dataset is multimodal. Defaults to False.

        mmap (bool): Whether to mmap the .bin files. Defaults to True.
    """

    def __init__(
        self,
        path_prefix: str,
        multimodal: bool = False,
        mmap: bool = True,
    ) -> None:
        super().__init__()
        self.path_prefix: str
        self.multimodal: bool
        self.mmap: bool

        self.bin_reader: _BinReader
        self.index: _IndexReader

        self.initialize(path_prefix, multimodal, mmap)

    def initialize(
        self,
        path_prefix: str,
        multimodal: bool,
        mmap: bool,
    ) -> None:
        """Initialize the dataset

        This method is called by IndexedDataset.__init__ during object creation and by
        IndexedDataset.__setstate__ during un-pickling

        Args:
            path_prefix (str): The index (.idx) and data (.bin) prefix

            multimodal (bool): Whether the dataset is multimodal

            mmap (bool): Whether to mmap the .bin file

            object_storage_config (Optional[ObjectStorageConfig]): See IndexedDataset docstring
                for details.
        """
        idx_path = get_idx_path(path_prefix)
        bin_path = get_bin_path(path_prefix)
        assert os.path.exists(idx_path) and os.path.exists(
            bin_path
        ), f"One or both of the .idx and .bin files cannot be found at the path prefix {path_prefix}"
        self.path_prefix = path_prefix
        self.multimodal = multimodal
        self.mmap = mmap
        if mmap:
            self.bin_reader = _MMapBinReader(bin_path)
        else:
            self.bin_reader = _FileBinReader(bin_path)
        self.index = _IndexReader(idx_path, self.multimodal)

    def __getstate__(self) -> tuple[str, bool, bool]:
        """Get the state during pickling

        Returns:
            tuple[str, bool, bool, Optional[ObjectStorageConfig]]: The state tuple
        """
        return self.path_prefix, self.multimodal, self.mmap

    def __setstate__(self, state: tuple[str, bool, bool]) -> None:
        """Set the state during un-pickling

        Args:
            state (tuple[str, bool, bool, Optional[ObjectStorageConfig]]): The state tuple
        """
        path_prefix, multimodal, mmap = state
        self.initialize(path_prefix, multimodal, mmap)

    def __del__(self) -> None:
        """Clean up the object"""
        del self.bin_reader
        del self.index

    def __len__(self) -> int:
        """Return the length of the dataset i.e. the number of sequences in the index

        Returns:
            int: The length of the dataset
        """
        return len(self.index)

    def __getitem__(
        self, index: Union[int, numpy.integer, slice]
    ) -> Union[
        numpy.ndarray,
        tuple[numpy.ndarray, numpy.number],
        list[numpy.ndarray],
        tuple[list[numpy.ndarray], numpy.ndarray],
    ]:
        """Return from the dataset

        Args:
            index (Union[int, numpy.integer, slice]): The index or index slice into the dataset

        Raises:
            ValueError: When the index slice is non-contiguous

            TypeError: When the index is of an unexpected type

        Returns:
            Union[
                numpy.ndarray,
                tuple[numpy.ndarray, numpy.number],
                list[numpy.ndarray],
                tuple[list[numpy.ndarray], numpy.ndarray],
            ]: The sequence tokens and modes at the index or index slice
        """
        if isinstance(index, (int, numpy.integer)):
            sequence_pointer, sequence_length, sequence_mode = self.index[index]
            sequence = self.bin_reader.read(
                dtype=self.index.dtype,
                count=int(sequence_length),
                offset=int(sequence_pointer),
            )
            return (sequence, sequence_mode) if sequence_mode is not None else sequence
        elif isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            sequence_lengths = self.index.sequence_lengths[index]
            assert self.index.sequence_modes is not None
            sequence_modes = (
                self.index.sequence_modes[index]
                if self.multimodal
                else None  # type: ignore[index]
            )
            sequence_offsets = list(accumulate(sequence_lengths))
            sequences = numpy.split(
                self.bin_reader.read(
                    dtype=self.index.dtype,
                    count=sum(sequence_lengths),
                    offset=self.index.sequence_pointers[start],
                ),
                sequence_offsets[:-1],
            )
            return (
                (sequences, sequence_modes) if sequence_modes is not None else sequences
            )
        else:
            raise TypeError(
                "Unexpected type received for index: {}".format(type(index))
            )

    def get(
        self, index: int, offset: int = 0, length: Optional[int] = None
    ) -> Union[numpy.ndarray, tuple[numpy.ndarray, numpy.number]]:
        """Retrieve a single item from the dataset with the option to only
        return a portion of the item.

        get(index) is the same as [index] but get() does not support slicing.

        Args:
            index (Union[int, numpy.integer]): The index into the dataset

            offset (int): The integer token offset in the sequence

            length (int): The number of tokens to grab from the sequence

        Returns:
            Union[numpy.ndarray, tuple[numpy.ndarray, numpy.number]]: The sequence tokens and mode
                at the index
        """
        sequence_pointer, sequence_length, sequence_mode = self.index[index]
        if length is None:
            length = int(sequence_length) - offset
        sequence_pointer += offset * DType.size(self.index.dtype)
        sequence = self.bin_reader.read(
            dtype=self.index.dtype, count=length, offset=int(sequence_pointer)
        )
        return (sequence, sequence_mode) if sequence_mode is not None else sequence

    @property
    def sequence_lengths(self) -> numpy.ndarray:
        """Get the sequence lengths

        Returns:
            numpy.ndarray: The sequence lengths
        """
        return self.index.sequence_lengths

    @property
    def document_indices(self) -> numpy.ndarray:
        """Get the document indices

        Returns:
            numpy.ndarray: The document indices
        """
        return self.index.document_indices

    def get_document_indices(self) -> numpy.ndarray:
        """Get the document indices

        This method is slated for deprecation.

        Returns:
            numpy.ndarray: The document indices
        """
        return self.index.document_indices

    def set_document_indices(self, document_indices: numpy.ndarray) -> None:
        """Set the document indices

        This method is slated for deprecation.

        Args:
            document_indices (numpy.ndarray): The document indices
        """
        self.index.document_indices = document_indices

    @property
    def sequence_modes(self) -> numpy.ndarray:
        """Get the sequence modes

        Returns:
            numpy.ndarray: The sequence modes
        """
        assert self.index.sequence_modes
        return self.index.sequence_modes

    @staticmethod
    def exists(path_prefix: str) -> bool:
        """Return whether the IndexedDataset exists on disk at the prefix

        Args:
            path_prefix (str): The prefix to the index (.idx) and data (.bin) files

        Returns:
            bool: Whether the IndexedDataset exists on disk at the prefix
        """

        return os.path.exists(get_idx_path(path_prefix)) and os.path.exists(
            get_bin_path(path_prefix)
        )
