from tokenizer import get_tokenizer
tokenizer = get_tokenizer("./tokenizer")
# tokenizer.encoder["<|endoftext|>"]
text_list = ["Traveling to a new country can be a life-changing experience.","The new policy had a disadvantageous impact on small businesses.","After careful reconsideration, they decided to change their plans.","Decentralization allows for greater autonomy and decision-making power at the local level."]
for text in text_list:
    print(text,"=====>")
    print([tokenizer.decode([i]) for i in tokenizer.encode(text)])