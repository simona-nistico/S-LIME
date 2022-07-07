
def logging(s, path, print_=True):
    if print_:
        print(s)
    if path:
        with open(path, 'a+') as f:
            f.write(s + '\n')

def load_sent(path):
    sents = []
    with open(path) as f:
        for line in f:
            # sents.append(line.split())
            sents.append(line)
    return sents