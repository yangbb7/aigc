from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
# 加载分词器
tokenizer = GPT2Tokenizer.from_pretrained("./sanguo")
tokenizer.add_special_tokens({
  "eos_token": "</s>",
  "bos_token": "<s>",
  "unk_token": "<unk>",
  "pad_token": "<pad>",
  "mask_token": "<mask>"
})
# 配置GPT2模型参数
config = GPT2Config(
  vocab_size=tokenizer.vocab_size,
  bos_token_id=tokenizer.bos_token_id,
  eos_token_id=tokenizer.eos_token_id
)
# 创建模型
model = GPT2LMHeadModel(config)
# 训练数据我们用按行分割
from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="./resource/sanguo.txt",
    block_size=32,
  # 如果训练时你的显存不够
  # 可以适当调小 block_size
)
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments
# 配置训练参数
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=20,
    per_gpu_train_batch_size=16,
    save_steps=2000,
    save_total_limit=2,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
trainer.train()
# 保存模型
model.save_pretrained('./sanguo')