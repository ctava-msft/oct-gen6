import json
import re

def parse_prompt(prompt):
    image_type_match = re.search(r'create an (\w+) image', prompt, re.IGNORECASE)
    layer_match = re.search(r'with a large (\w+)', prompt, re.IGNORECASE)
    
    data = {}
    if image_type_match:
        data['imageType'] = image_type_match.group(1)
    if layer_match:
        data['layer'] = layer_match.group(1)
        data['changeSize'] = 'large'
    return data

def main():
    prompt = "create an OCT image with a large RNFL."
    extracted_data = parse_prompt(prompt)
    
    with open('output.json', 'w') as json_file:
        json.dump(extracted_data, json_file, indent=4)

if __name__ == "__main__":
    main()