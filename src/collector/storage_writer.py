"""
Storage Writer — Decision Record Persistence
=============================================

Persists decision records to JSONL files with batching and rotation.

Features:
- Batching: collects N records before writing
- Periodic flush: writes every M seconds even if batch not full
- File rotation: by date or market
- Atomic writes: rename after complete

Usage:
    writer = StorageWriter(
        output_dir="/data/decisions",
        batch_size=100,
        flush_interval_sec=5,
    )
    
    # In decision loop:
    writer.write(record)
    
    # On shutdown:
    writer.close()

File naming:
    decisions_YYYY-MM-DD_HHMMSS.jsonl
    decisions_YYYY-MM-DD_HHMMSS_part2.jsonl (on rotation)
"""

import json
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class StorageWriter:
    """
    JSONL storage writer with batching and rotation.
    
    Args:
        output_dir: Directory for output files
        batch_size: Records per batch before write (default: 100)
        flush_interval_sec: Force flush interval (default: 5)
        max_file_size_mb: Rotate file after this size (default: 100)
        prefix: File name prefix (default: "decisions")
        
    Thread Safety:
        All methods are thread-safe via internal lock.
    """
    
    def __init__(
        self,
        output_dir: str = "./data/decisions",
        batch_size: int = 100,
        flush_interval_sec: float = 5.0,
        max_file_size_mb: float = 100.0,
        prefix: str = "decisions",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.flush_interval_sec = flush_interval_sec
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.prefix = prefix
        
        # State
        self._lock = threading.Lock()
        self._buffer: deque[dict] = deque()
        self._current_file: Optional[Path] = None
        self._current_file_handle: Optional[Any] = None
        self._current_file_size: int = 0
        self._last_flush_time: float = time.time()
        self._file_part: int = 1
        self._current_date: Optional[str] = None
        
        # Statistics
        self._total_written: int = 0
        self._total_batches: int = 0
        self._total_files: int = 0
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "storage_writer_initialized",
            extra={
                "output_dir": str(self.output_dir),
                "batch_size": batch_size,
                "flush_interval_sec": flush_interval_sec,
                "max_file_size_mb": max_file_size_mb,
            },
        )
    
    def write(self, record: dict) -> None:
        """
        Add record to buffer and flush if needed.
        
        Args:
            record: Decision record to write
        """
        with self._lock:
            self._buffer.append(record)
            
            # Check if we should flush
            should_flush = (
                len(self._buffer) >= self.batch_size or
                time.time() - self._last_flush_time >= self.flush_interval_sec
            )
            
            if should_flush:
                self._flush_locked()
    
    def flush(self) -> int:
        """
        Force flush buffer to disk.
        
        Returns:
            Number of records written
        """
        with self._lock:
            return self._flush_locked()
    
    def _flush_locked(self) -> int:
        """Flush buffer (must hold lock)."""
        if not self._buffer:
            self._last_flush_time = time.time()
            return 0
        
        # Check for date rotation
        current_date = datetime.utcnow().strftime("%Y-%m-%d")
        if self._current_date != current_date:
            self._rotate_file_locked(current_date)
        
        # Ensure file is open
        if self._current_file_handle is None:
            self._open_new_file_locked(current_date)
        
        # Write records
        records_written = 0
        while self._buffer:
            record = self._buffer.popleft()
            
            # Serialize
            try:
                line = json.dumps(record, default=str) + "\n"
                line_bytes = line.encode("utf-8")
            except Exception as e:
                logger.warning(
                    "storage_serialize_error",
                    extra={"error": str(e)},
                )
                continue
            
            # Check file size rotation
            if self._current_file_size + len(line_bytes) > self.max_file_size_bytes:
                self._rotate_file_locked(current_date, size_rotation=True)
            
            # Write
            try:
                self._current_file_handle.write(line)
                self._current_file_size += len(line_bytes)
                records_written += 1
                self._total_written += 1
            except Exception as e:
                logger.error(
                    "storage_write_error",
                    extra={"error": str(e)},
                )
                # Put record back
                self._buffer.appendleft(record)
                break
        
        # Flush to disk
        if self._current_file_handle:
            try:
                self._current_file_handle.flush()
                os.fsync(self._current_file_handle.fileno())
            except Exception as e:
                logger.warning(
                    "storage_fsync_error",
                    extra={"error": str(e)},
                )
        
        self._last_flush_time = time.time()
        self._total_batches += 1
        
        if records_written > 0:
            logger.debug(
                "storage_flushed",
                extra={
                    "records": records_written,
                    "file": str(self._current_file),
                    "file_size_kb": round(self._current_file_size / 1024, 1),
                },
            )
        
        return records_written
    
    def _open_new_file_locked(self, date_str: str) -> None:
        """Open a new output file (must hold lock)."""
        # Close existing file
        if self._current_file_handle:
            self._current_file_handle.close()
            self._current_file_handle = None
        
        # Generate filename
        timestamp = datetime.utcnow().strftime("%H%M%S")
        if self._file_part > 1:
            filename = f"{self.prefix}_{date_str}_{timestamp}_part{self._file_part}.jsonl"
        else:
            filename = f"{self.prefix}_{date_str}_{timestamp}.jsonl"
        
        self._current_file = self.output_dir / filename
        self._current_date = date_str
        
        # Open file
        self._current_file_handle = open(self._current_file, "a", encoding="utf-8")
        self._current_file_size = self._current_file.stat().st_size if self._current_file.exists() else 0
        self._total_files += 1
        
        logger.info(
            "storage_file_opened",
            extra={
                "file": str(self._current_file),
                "part": self._file_part,
            },
        )
    
    def _rotate_file_locked(
        self,
        date_str: str,
        size_rotation: bool = False,
    ) -> None:
        """Rotate to new file (must hold lock)."""
        # Close current file
        if self._current_file_handle:
            self._current_file_handle.close()
            self._current_file_handle = None
            
            logger.info(
                "storage_file_rotated",
                extra={
                    "old_file": str(self._current_file),
                    "reason": "size" if size_rotation else "date",
                    "size_kb": round(self._current_file_size / 1024, 1),
                },
            )
        
        # Update part number
        if size_rotation and self._current_date == date_str:
            self._file_part += 1
        else:
            self._file_part = 1
        
        # Open new file
        self._open_new_file_locked(date_str)
    
    def close(self) -> None:
        """Close writer and flush remaining records."""
        with self._lock:
            # Flush remaining
            self._flush_locked()
            
            # Close file
            if self._current_file_handle:
                self._current_file_handle.close()
                self._current_file_handle = None
                
                logger.info(
                    "storage_writer_closed",
                    extra={
                        "total_written": self._total_written,
                        "total_batches": self._total_batches,
                        "total_files": self._total_files,
                    },
                )
    
    def get_stats(self) -> dict:
        """Get writer statistics."""
        with self._lock:
            return {
                "total_written": self._total_written,
                "total_batches": self._total_batches,
                "total_files": self._total_files,
                "buffer_size": len(self._buffer),
                "current_file": str(self._current_file) if self._current_file else None,
                "current_file_size_kb": round(self._current_file_size / 1024, 1),
                "last_flush_ago_sec": round(time.time() - self._last_flush_time, 1),
            }
    
    def snapshot(self) -> dict:
        """Full snapshot for API."""
        return self.get_stats()
