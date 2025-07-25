"""
Resource Management System
Manages temporary files, GPU memory, and cleanup operations for RunPod optimization
"""

import os
import shutil
import tempfile
import logging
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional
import psutil
import torch
import gc

logger = logging.getLogger(__name__)

class ResourceManager:
    """Manages system resources and cleanup operations"""
    
    def __init__(self, temp_dir: str = None, cleanup_interval: int = 3600):
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.cleanup_interval = cleanup_interval
        self.active_jobs = {}
        self.cleanup_thread = None
        self.stop_cleanup = False
        
        # Resource limits
        self.max_temp_size = 5 * 1024 * 1024 * 1024  # 5GB
        self.max_gpu_memory = 0.8  # 80% of available GPU memory
        self.cleanup_threshold = 0.7  # 70% of max temp size
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def register_job(self, job_id: str, job_dir: str, output_dir: str):
        """Register a new job for tracking"""
        self.active_jobs[job_id] = {
            'job_dir': job_dir,
            'output_dir': output_dir,
            'created_at': time.time(),
            'last_accessed': time.time(),
            'size': 0
        }
        logger.info(f"Registered job {job_id} for resource tracking")
    
    def update_job_access(self, job_id: str):
        """Update job access time"""
        if job_id in self.active_jobs:
            self.active_jobs[job_id]['last_accessed'] = time.time()
    
    def cleanup_job(self, job_id: str, force: bool = False):
        """Clean up resources for a specific job"""
        if job_id not in self.active_jobs:
            logger.warning(f"Job {job_id} not found in active jobs")
            return False
        
        job_info = self.active_jobs[job_id]
        
        try:
            # Clean up job directory
            if os.path.exists(job_info['job_dir']):
                self._safe_remove_directory(job_info['job_dir'])
                logger.info(f"Cleaned up job directory for {job_id}")
            
            # Clean up output directory (keep for some time)
            if force or self._should_cleanup_output(job_info):
                if os.path.exists(job_info['output_dir']):
                    self._safe_remove_directory(job_info['output_dir'])
                    logger.info(f"Cleaned up output directory for {job_id}")
            
            # Remove from active jobs
            del self.active_jobs[job_id]
            
            # Force garbage collection
            gc.collect()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup job {job_id}: {e}")
            return False
    
    def cleanup_all_jobs(self, max_age_hours: int = 24):
        """Clean up all jobs older than specified age"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        jobs_to_cleanup = []
        
        for job_id, job_info in self.active_jobs.items():
            age = current_time - job_info['created_at']
            if age > max_age_seconds:
                jobs_to_cleanup.append(job_id)
        
        logger.info(f"Cleaning up {len(jobs_to_cleanup)} old jobs")
        
        for job_id in jobs_to_cleanup:
            self.cleanup_job(job_id, force=True)
    
    def cleanup_temp_files(self):
        """Clean up temporary files and directories"""
        try:
            temp_path = Path(self.temp_dir)
            
            # Find temporary files created by our application
            temp_files = []
            for pattern in ['avatar_*', 'selfies_*', 'output_*', '*.tmp']:
                temp_files.extend(temp_path.glob(pattern))
            
            # Remove old temporary files
            current_time = time.time()
            for temp_file in temp_files:
                try:
                    if temp_file.is_file():
                        file_age = current_time - temp_file.stat().st_mtime
                        if file_age > 3600:  # 1 hour
                            temp_file.unlink()
                            logger.debug(f"Removed old temp file: {temp_file}")
                    elif temp_file.is_dir():
                        dir_age = current_time - temp_file.stat().st_mtime
                        if dir_age > 7200:  # 2 hours
                            shutil.rmtree(temp_file, ignore_errors=True)
                            logger.debug(f"Removed old temp directory: {temp_file}")
                except Exception as e:
                    logger.debug(f"Failed to remove temp file {temp_file}: {e}")
            
            logger.info("Temporary files cleanup completed")
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {e}")
    
    def cleanup_gpu_memory(self):
        """Clean up GPU memory"""
        try:
            if torch.cuda.is_available():
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Get GPU memory info
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory['allocated_bytes.all.current'] / 1024**3  # GB
                reserved = gpu_memory['reserved_bytes.all.current'] / 1024**3  # GB
                
                logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
                
                # Force garbage collection
                gc.collect()
                
                # Clear cache again
                torch.cuda.empty_cache()
                
                logger.info("GPU memory cleanup completed")
            else:
                logger.info("GPU not available, skipping GPU cleanup")
                
        except Exception as e:
            logger.error(f"Failed to cleanup GPU memory: {e}")
    
    def get_system_status(self) -> Dict:
        """Get current system resource status"""
        try:
            # CPU and memory info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage(self.temp_dir)
            
            # GPU info
            gpu_info = {}
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_stats()
                gpu_info = {
                    'allocated_gb': gpu_memory['allocated_bytes.all.current'] / 1024**3,
                    'reserved_gb': gpu_memory['reserved_bytes.all.current'] / 1024**3,
                    'device_count': torch.cuda.device_count()
                }
            
            # Active jobs info
            active_jobs_count = len(self.active_jobs)
            total_job_size = sum(job['size'] for job in self.active_jobs.values())
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / 1024**3,
                'disk_percent': (disk.used / disk.total) * 100,
                'disk_available_gb': disk.free / 1024**3,
                'gpu': gpu_info,
                'active_jobs': active_jobs_count,
                'total_job_size_gb': total_job_size / 1024**3,
                'temp_dir_size_gb': self._get_directory_size(self.temp_dir) / 1024**3
            }
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {}
    
    def check_resource_limits(self) -> Dict:
        """Check if system is approaching resource limits"""
        status = self.get_system_status()
        warnings = []
        
        # Check memory usage
        if status.get('memory_percent', 0) > 80:
            warnings.append(f"High memory usage: {status['memory_percent']:.1f}%")
        
        # Check disk usage
        if status.get('disk_percent', 0) > 80:
            warnings.append(f"High disk usage: {status['disk_percent']:.1f}%")
        
        # Check GPU memory
        if status.get('gpu', {}).get('allocated_gb', 0) > 8:  # 8GB threshold
            warnings.append(f"High GPU memory usage: {status['gpu']['allocated_gb']:.2f}GB")
        
        # Check temp directory size
        temp_size_gb = status.get('temp_dir_size_gb', 0)
        if temp_size_gb > 3:  # 3GB threshold
            warnings.append(f"Large temp directory: {temp_size_gb:.2f}GB")
        
        return {
            'warnings': warnings,
            'needs_cleanup': len(warnings) > 0,
            'status': status
        }
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while not self.stop_cleanup:
                try:
                    # Check resource limits
                    limits = self.check_resource_limits()
                    
                    if limits['needs_cleanup']:
                        logger.info("Resource limits exceeded, performing cleanup")
                        
                        # Cleanup old jobs
                        self.cleanup_all_jobs(max_age_hours=1)
                        
                        # Cleanup temp files
                        self.cleanup_temp_files()
                        
                        # Cleanup GPU memory
                        self.cleanup_gpu_memory()
                        
                        logger.info("Cleanup completed")
                    
                    # Sleep for cleanup interval
                    time.sleep(self.cleanup_interval)
                    
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
                    time.sleep(60)  # Wait 1 minute on error
        
        self.cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info("Resource cleanup thread started")
    
    def _safe_remove_directory(self, directory: str):
        """Safely remove directory with error handling"""
        try:
            if os.path.exists(directory):
                shutil.rmtree(directory, ignore_errors=True)
                logger.debug(f"Removed directory: {directory}")
        except Exception as e:
            logger.warning(f"Failed to remove directory {directory}: {e}")
    
    def _should_cleanup_output(self, job_info: Dict) -> bool:
        """Determine if output directory should be cleaned up"""
        current_time = time.time()
        age_hours = (current_time - job_info['created_at']) / 3600
        
        # Clean up outputs older than 24 hours
        return age_hours > 24
    
    def _get_directory_size(self, directory: str) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception as e:
            logger.warning(f"Failed to get directory size for {directory}: {e}")
        
        return total_size
    
    def shutdown(self):
        """Shutdown resource manager"""
        logger.info("Shutting down resource manager")
        self.stop_cleanup = True
        
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        
        # Final cleanup
        self.cleanup_all_jobs(force=True)
        self.cleanup_temp_files()
        self.cleanup_gpu_memory()
        
        logger.info("Resource manager shutdown completed") 