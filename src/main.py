from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='./sanguo')
set_seed(42)
txt = generator("吕布", max_length=10)
print(txt)

txt = generator("接着奏乐", max_length=10)
print(txt)


txt = generator("三国", max_length=1000)
print(txt)