# src/kmer_ord/workflow/operation.py

from abc import ABC, abstractmethod


class Operation(ABC):
    name = None
    requires = []
    produces = []

    @abstractmethod
    def run(self, context):
        pass