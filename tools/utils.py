

def read_corpus(filename):
    content = []
    with open(filename) as f:
        for idx, line in enumerate(f):
            content.append(line)
    return content
