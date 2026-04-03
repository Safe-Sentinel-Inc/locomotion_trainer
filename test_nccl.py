"""Quick NCCL distributed test."""
import os
import torch
import torch.distributed as dist

local_rank = int(os.environ.get("LOCAL_RANK", "0"))
rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

torch.cuda.set_device(local_rank)
print(f"[Rank {rank}] Starting init_process_group on cuda:{local_rank}", flush=True)

dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
print(f"[Rank {rank}] init_process_group done", flush=True)

# Test all_reduce
t = torch.tensor([rank], dtype=torch.float32, device=f"cuda:{local_rank}")
dist.all_reduce(t)
print(f"[Rank {rank}] all_reduce result: {t.item()} (expected {sum(range(world_size))})", flush=True)

# Test broadcast_object_list (what rsl_rl uses)
objects = [{"key": f"from_rank_{rank}"}] if rank == 0 else [None]
dist.broadcast_object_list(objects, src=0)
print(f"[Rank {rank}] broadcast_object_list result: {objects}", flush=True)

dist.destroy_process_group()
print(f"[Rank {rank}] DONE", flush=True)
