.PHONY: train-mapping train-mapping-legacy

# MappingNet pretraining (Paper Phase 0)
train-mapping:
	python scripts/train_mapping.py --use_warp --num_steps 50000 --batch_size 64

train-mapping-legacy:
	python scripts/train_mapping.py --no_warp --num_steps 50000 --batch_size 64
