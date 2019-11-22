class FetchException(Exception):
    def __init__(self, message):
        super(Exception, self).__init__(message)


class PreprocessException(Exception):
    def __init__(self, message):
        super(Exception, self).__init__(message)


class NoDataException(Exception):
    def __init__(self, message):
        super(Exception, self).__init__(message)
