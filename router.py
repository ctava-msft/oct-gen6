import spacy
import json

# Load spaCy's English language model
nlp = spacy.load('en_core_web_sm')

# The prompt
prompt = "create an OCT image with an large RNFL"

# Process the prompt
doc = nlp(prompt)

# Define possible values
image_types = ['OCT', 'Fundus', 'Fluorescein Angiography']
layers = ['RNFL', 'GCL', 'IPL', 'INL', 'OPL', 'ONL', 'IS', 'OS', 'RPE']
change_sizes = ['small', 'medium', 'large']

# Initialize variables
imageType = None
layer = None
changeSize = None

# Extract entities from the prompt
for token in doc:
    word = token.text
    if word in image_types:
        imageType = word
    if word in layers:
        layer = word
    if word in change_sizes:
        changeSize = word

# Create the result dictionary
result = {
    'imageType': imageType,
    'layer': layer,
    'changeSize': changeSize
}

# Output to JSON file
with open('result.json', 'w') as f:
    json.dump(result, f, indent=4)

print("Result saved to result.json:")
print(json.dumps(result, indent=4))
