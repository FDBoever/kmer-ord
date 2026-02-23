# src/kmer_ord/workflow/runner.py
class Runner:
    def __init__(self, operations):
        self.operations = operations

    def run(self, context):
        for op in self.operations:
            for requirement in getattr(op, "requires", []):
                context.get(requirement)

            context.logger.info(f"Running operation: {op.name}")
            op.run(context)