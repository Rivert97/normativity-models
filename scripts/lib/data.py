"""Module to define classes to store different types of documents."""

import os

import numpy as np

class Document:
    """Class to store the information of a chunk of text."""

    def __init__(self, content:str, metadata:dict=None, embeddings:np.array=None,
                 distance:float=None):
        self.content = content
        self.metadata = metadata
        self.embeddings = embeddings
        self.distance = distance

    def get_content(self):
        """Get the text contained in the document."""
        return self.content

    def get_metadata(self):
        """Get the metadata associated with the document."""
        return self.metadata

    def get_embeddings(self):
        """Get the embeddings array."""
        return self.embeddings

    def get_distance(self):
        """Get the distance between the query and the document."""
        return self.distance

    def get_reference(self):
        """Return a string with a human readable reference to the document."""
        ref = self.metadata['document_name']
        ref += ': ' + ', '.join(self.metadata['path'].strip("'").split('/')[2:])

        return ref

    def get_for_standard_pipeline(self):
        """Return a dictionary formated for Hugging Face standard 'rag' pipeline."""
        doc = {
            'title': self.metadata['title'],
            'text': self.content,
        }

        return doc

    def print_to_console(self):
        """Print the document and its reference in a human readable format in console."""
        c_width, _ = os.get_terminal_size()

        reference = self.get_reference()
        title = self.get_metadata()['title']
        content = self.get_content()

        self.__print_box(reference + "\n\n" + title + "\n\n" + content, c_width)

    def __print_box(self, text:str, width:int):
        print('+', '-'*(width-2), '+', sep='')
        print('| ', end='')
        i = 2
        for t in text:
            if t == '\n':
                print(' '*(width - i - 2), '|')
                print('| ', end='')
                i = 2
            elif i < width - 2:
                print(t, end='')
                i += 1
            else:
                print(' |')
                print('|', t, end='')
                i = 3
        print(' '*(width-i-2), '|')
        print('+', '-'*(width-2), '+', sep='')

    def __str__(self):
        if self.metadata:
            metadata_str = ', '.join([f"{key}:{value}" for key, value in self.metadata.items()])
        else:
            metadata_str = 'None'
        return f"Document(content='{self.content}', metadata={{{metadata_str}}})"
