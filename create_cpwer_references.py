import xml.etree.ElementTree as ET
import os
import argparse

def main(input_dir, output_dir):
    words_dir = os.path.join(input_dir, "words")
    for filename in os.listdir(words_dir):
        if not filename.endswith('.words.xml'):
            continue
        xml_path = os.path.join(words_dir, filename)
        with open(xml_path, 'r') as file:
            xml_data = file.read()

        root = ET.fromstring(xml_data)

        transcript = ""

        for word in root.iter('w'):
            if word.attrib.get('punc') == 'true':
                transcript += word.text
            else:
                if transcript == "":
                    transcript = word.text
                else:
                    transcript = transcript + " " + word.text
            
        destination = os.path.join(output_dir, filename.replace(".words.xml", ".words_joined.txt"))
        with open(destination, 'w') as file:
            file.write(transcript)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir', help='Input directory', required=True)
    parser.add_argument('-o', '--output_dir', help='Output directory', required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
