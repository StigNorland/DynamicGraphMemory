import tiktoken

enc = tiktoken.get_encoding('gpt2')

words = ['strawberry', 'STRAWBERRY', 'retrieval', 'environment', 'mutation']

for word in words:
    tokens = enc.encode(word)
    decoded = [enc.decode([t]) for t in tokens]
    print(f'{word:20s} -> {decoded}')