import os

p = "DataProcessor/TextProcessor/src/extractors"

for file in os.listdir(p):
    if not os.path.isdir(f"{p}/{file}"):
        extractor = file.replace(".py", '')
        os.makedirs(f"{p}/{extractor}")