CUDA_VISIBLE_DEVICES=0 accelerate launch fine_tune_secondStage.py \
  --pretrained_model_name_or_path="/path_stable-diffusion-3-medium-diffusers"  \
  --pretrained_lora_weights="/lora_weight_path" \
  --instance_data_dir="/range_image_data_path" \
  --output_dir="./output_path" \
  --mixed_precision="fp16" \
  --instance_prompt="A range image" \
  --resolution=1024 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=1000 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --rank=16 \
  --max_train_steps=10000 \
  --validation_prompt="A range image" \
  --validation_epochs=1 \
  --seed="0" \
  --height=256 \
  --width=1024 \
  --enable_3d_loss \
  --enable_edge_loss

CUDA_VISIBLE_DEVICES=0 accelerate launch generate_pc.py
