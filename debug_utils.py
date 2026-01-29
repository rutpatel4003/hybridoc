import torch
import gc
import subprocess
from datetime import datetime

def get_nvidia_smi_memory():
    """Get GPU memory from nvidia-smi (actual system view, not just PyTorch)."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            used, total = result.stdout.strip().split(',')
            return int(used.strip()), int(total.strip())
    except Exception:
        pass
    return None, None

def log_gpu_memory(tag: str):
    """Log current GPU memory usage with a tag.
    Shows both PyTorch tracked memory AND nvidia-smi actual memory.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        # also get nvidia-smi view
        smi_used, smi_total = get_nvidia_smi_memory()
        if smi_used is not None:
            smi_used_gb = smi_used / 1024
            print(f"📉 MEM [{datetime.now().strftime('%H:%M:%S')}] {tag}: "
                  f"Torch={allocated:.2f}GB | nvidia-smi={smi_used_gb:.2f}GB")
        else:
            print(f"📉 MEM [{datetime.now().strftime('%H:%M:%S')}] {tag}: Alloc={allocated:.2f}GB, Rsrv={reserved:.2f}GB")
    else:
         print(f"📉 MEM [{datetime.now().strftime('%H:%M:%S')}] {tag}: CUDA not available")

def explicit_cleanup(tag: str = ""):
    """Force GC and empty cache."""
    if tag:
        print(f"Cleanup triggered: {tag}")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_gpu_memory(f"After Cleanup ({tag})")

