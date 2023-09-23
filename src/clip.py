from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from transformers import GPT2TokenizerFast

# 构建分词器 GPT2 基于 BPE 算法实现
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.normalizer = Sequence([NFKC()])
tokenizer.pre_tokenizer = ByteLevel()
tokenizer.decoder = ByteLevelDecoder()

special_tokens = ["<s>","<pad>","</s>","<unk>","<mask>"]
trainer = BpeTrainer(vocab_size=50000, show_progress=True, inital_alphabet=ByteLevel.alphabet(), special_tokens=special_tokens)

files = ["resource/sanguo.txt"]
# 开始训练了
tokenizer.train(files, trainer)
# 把训练的分词通过GPT2保存起来，以方便后续使用
newtokenizer = GPT2TokenizerFast(tokenizer_object=tokenizer)
newtokenizer.save_pretrained("./sanguo")