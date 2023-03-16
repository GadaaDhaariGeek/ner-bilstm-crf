from sklearn.base import BaseEstimator, TransformerMixin

class MemoryTagger:

    def __init__(self):
        self.vocab = {}
        self.tags = []
        self.memory = {}

    def fit(self, X, y):
        """ Fit the model on data

        Args:
            X (list): 2D list having sentences
            y (list): 2D list having tags
        """
        for sentence, corr_tags in zip(X, y):
            for word, tag in zip(sentence, corr_tags):
                # print(word, tag)
                # print(type(word), type(tag))
                if tag not in self.tags:
                    self.tags.append(tag)
                if word in self.vocab:
                    if tag in self.vocab[word]:
                        self.vocab[word][tag] += 1
                    else:
                        self.vocab[word][tag] = 1
                else:
                    self.vocab[word] = {tag: 1}
            
        for key, value_dict in self.vocab.items():
            # for a particular vocabulary word, give me the tag which has maximum support
            self.memory[key] = max(value_dict, key=value_dict.get)

    def predict(self, X):
        """Method to do the prerdictions 

        Args:
            X (list): 2D list having sentences
        """
        return [[self.memory.get(word, "O") for word in sentence] for sentence in X]