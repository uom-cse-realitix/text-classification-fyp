import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')


class MemoryAnalyzer:

    def __init__(self, name):
        self.name = name

    @staticmethod
    def analyze(dataframe):
        for dtype in ['float', 'int', 'object']:
            selected_dtype = dataframe.select_dtypes(include=[dtype])
            mean_usage_bytes = selected_dtype.memory_usage(deep=True).mean()
            mean_usage_mega_bytes = mean_usage_bytes / (1024 * 1024)
            logging.info("Average memory usage for {} column: {:03.2f} MB".format(dtype, mean_usage_mega_bytes))
