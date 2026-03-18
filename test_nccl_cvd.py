"""Test NCCL with CUDA_VISIBLE_DEVICES masking (each rank sees 1 GPU)."""
import os
import torch
import torch.distributed as dist

rank = int(os.environ.get("RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))

# Each rank sees only its own GPU
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
device = "cuda:0"
torch.cuda.set_device(0)

print(f"[Rank {rank}] CVD={os.environ['CUDA_VISIBLE_DEVICES']}, device={device}", flush=True)

dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
print(f"[Rank {rank}] init_process_group done", flush=True)

t = torch.tensor([rank], dtype=torch.float32, device=device)
dist.all_reduce(t)
print(f"[Rank {rank}] all_reduce result: {t.item()} (expected {sum(range(world_size))})", flush=True)

objects = [{"key": f"from_rank_{rank}"}] if rank == 0 else [None]
dist.broadcast_object_list(objects, src=0)
print(f"[Rank {rank}] broadcast OK: {objects}", flush=True)

dist.destroy_process_group()
print(f"[Rank {rank}] DONE", flush=True)
