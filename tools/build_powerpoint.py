"""Build the editable PowerPoint and synchronized storyboard from local evidence."""
from presentation_design import build
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Alternate output PPTX when the main deck is open.')
    build(parser.parse_args().output)
